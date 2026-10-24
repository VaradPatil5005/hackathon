"""
app.py - Smart Waste Segregation Assistant (VIT AI Innovate-A-Thon, Question 6)
Run:  streamlit run app.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from waste_core import (
    ARTIFACTS,
    CATEGORIES,
    CATEGORY_NAMES,
    CLASS_INFO,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_ENTROPY_THRESHOLD,
    MODEL_PATH,
    ROOT,
    UNCERTAIN_ADVICE,
    aggregate_to_categories,
    apply_perturbations,
    gate,
    load_trained_model,
    normalized_entropy,
    predict_probs,
)

DEMO_DIR = ROOT / "demo_samples"

st.set_page_config(
    page_title="Smart Waste Segregation Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .bin-card {padding: 18px 22px; border-radius: 12px; margin-bottom: 14px;}
    .bin-card h2 {margin: 0 0 6px 0; font-size: 1.6rem;}
    .bin-card p {margin: 2px 0; font-size: 1.02rem;}
    .warn-card {padding: 18px 22px; border-radius: 12px; background: #FFF3E0; border-left: 8px solid #EF6C00;
                color: #3E2723; margin-bottom: 14px;}
    .warn-card h2 {margin: 0 0 6px 0; font-size: 1.45rem; color: #E65100;}
    .small-note {font-size: 0.85rem; opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the waste classifier...")
def get_model():
    return load_trained_model()


@st.cache_data
def load_json(name: str):
    path = ARTIFACTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_split():
    path = ARTIFACTS / "split.csv"
    return pd.read_csv(path) if path.exists() else None


def demo_samples():
    if not DEMO_DIR.exists():
        return {}
    return {p.stem.replace("_", " "): p for p in sorted(DEMO_DIR.glob("*.jpg"))}


def synthetic_pattern(kind: str) -> Image.Image:
    rng = np.random.default_rng(7)
    if kind == "Random noise":
        return Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    if kind == "Empty grey tray":
        return Image.fromarray(np.full((224, 224, 3), 128, np.uint8))
    cb = ((np.indices((224, 224)) // 28).sum(0) % 2 * 230).astype(np.uint8)
    return Image.fromarray(np.stack([cb] * 3, -1))


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found at `artifacts/model.pt`.\n\n"
        "Run `python train.py` and `python evaluate.py` first (see README.md)."
    )
    st.stop()

model, CLASS_NAMES, MODEL_META, DEVICE = get_model()
REPORT = load_json("evaluation_report.json")
SPLIT = load_json("split_summary.json")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("**Uncertainty / out-of-scope gate**")
    conf_t = st.slider("Minimum confidence (τ_conf)", 0.30, 0.95, DEFAULT_CONF_THRESHOLD, 0.05,
                       help="If the top class probability is below this, the app asks the user for help.")
    ent_t = st.slider("Maximum normalized entropy (τ_ent)", 0.10, 0.90, DEFAULT_ENTROPY_THRESHOLD, 0.05,
                      help="High entropy = probability spread over many classes = the model is confused.")
    st.divider()
    st.markdown("**Lighting & occlusion simulator**")
    st.caption("Degrade the image live to show how predictions and the safety gate react.")
    gamma = st.slider("Lighting gamma (0.3 dark → 2.5 overexposed)", 0.3, 2.5, 1.0, 0.1)
    glare = st.slider("Overhead glare", 0.0, 1.0, 0.0, 0.1)
    occlusion = st.slider("Hand / napkin occlusion (% area)", 0, 70, 0, 5)
    blur = st.slider("Motion / focus blur", 0, 10, 0, 1)
    st.divider()
    st.caption(
        f"Model: {MODEL_META.get('architecture', 'ConvNeXt-Tiny')}  \n"
        f"Running on: **{DEVICE.upper()}**  \n"
        f"Images are processed in memory only and never saved."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("♻️ Smart Waste Segregation Assistant")
st.markdown(
    "Point a camera at a cafeteria waste item → the assistant tells you **which bin** it goes in and **how to "
    "dispose of it**. If it is not sure, it **says so** and asks you to check the label or ask staff."
)
if REPORT:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy (5 bins)", f"{REPORT['category']['accuracy']:.1%}")
    c2.metric("Test accuracy (12 classes)", f"{REPORT['fine_grained']['accuracy']:.1%}")
    c3.metric("Test images (never seen in training)", f"{REPORT['test_size']:,}")
    if "imagenette" in REPORT.get("ood", {}):
        c4.metric("Non-waste images flagged", f"{REPORT['ood']['imagenette']['flag_rate']:.0%}")

tab_scan, tab_eval, tab_light, tab_arch = st.tabs([
    "📷 Live Scanner",
    "📊 Dataset & Evaluation",
    "💡 Lighting & Occlusion",
    "🏗️ Architecture, Responsible AI & SDGs",
])

# ===========================================================================
# TAB 1 - SCANNER
# ===========================================================================
with tab_scan:
    col_in, col_out = st.columns([1, 1.15], gap="large")
    raw = None
    with col_in:
        st.subheader("1 · Provide an image")
        mode = st.radio("Input source", ["📤 Upload", "📸 Camera", "🗂️ Demo samples", "🧪 Out-of-scope test"],
                        horizontal=True, label_visibility="collapsed")
        if mode == "📤 Upload":
            up = st.file_uploader("Upload a photo of one waste item", type=["jpg", "jpeg", "png", "webp", "bmp"])
            if up is not None:
                try:
                    raw = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
                except Exception as exc:  # corrupted / unsupported file
                    st.error(f"Could not read this image: {exc}")
        elif mode == "📸 Camera":
            cam = st.camera_input("Hold one item in front of the camera, fill the frame")
            if cam is not None:
                raw = ImageOps.exif_transpose(Image.open(cam)).convert("RGB")
        elif mode == "🗂️ Demo samples":
            samples = demo_samples()
            if samples:
                choice = st.selectbox("Demo image", list(samples))
                raw = Image.open(samples[choice]).convert("RGB")
            else:
                st.warning("No demo samples yet - run `python generate_demo_samples.py`.")
        else:
            kind = st.selectbox("Pattern", ["Random noise", "Empty grey tray", "Checkerboard"])
            raw = synthetic_pattern(kind)

        if raw is not None:
            raw.thumbnail((1024, 1024))
            img = apply_perturbations(raw, gamma=gamma, glare=glare, occlusion_pct=occlusion, blur=blur)
            perturbed = gamma != 1.0 or glare > 0 or occlusion > 0 or blur > 0
            st.image(img, use_container_width=True,
                     caption="Image as seen by the model" + (" (with simulated degradation)" if perturbed else ""))
        else:
            st.info("Upload a photo, take a camera snapshot, or pick a demo sample.")

    with col_out:
        st.subheader("2 · Where does it go?")
        if raw is not None:
            t0 = time.perf_counter()
            probs = predict_probs(model, [img], DEVICE)[0]
            latency = (time.perf_counter() - t0) * 1000
            top = int(np.argmax(probs))
            cls = CLASS_NAMES[top]
            info = CLASS_INFO[cls]
            cat = info["category"]
            msp, h = float(probs[top]), float(normalized_entropy(probs))
            uncertain, reasons = gate(msp, h, conf_t, ent_t)

            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence (MSP)", f"{msp:.1%}")
            m2.metric("Normalized entropy", f"{h:.3f}")
            m3.metric("Latency", f"{latency:.0f} ms")

            if uncertain:
                st.markdown(
                    f"""<div class="warn-card"><h2>⚠️ Not sure - please check</h2>
                    <p>The assistant cannot confidently classify this item ({'; '.join(reasons)}).
                    It may be out of scope, blocked by a hand, or badly lit.</p></div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(UNCERTAIN_ADVICE)
                st.caption(f"Best tentative guess (not trusted): {info['display']} → {cat} ({msp:.0%})")
                st.markdown("**Checked the label? Tell us what it is:**")
                manual = st.selectbox("I checked - this item is", ["-- choose --"] + [
                    CLASS_INFO[c]["display"] for c in CLASS_NAMES], key="manual")
                if manual != "-- choose --":
                    mcls = next(c for c in CLASS_NAMES if CLASS_INFO[c]["display"] == manual)
                    minfo = CLASS_INFO[mcls]
                    st.success(f"**{CATEGORIES[minfo['category']]['icon']} {minfo['category']} → "
                               f"{minfo['target']}**\n\n{minfo['instruction']}")
            else:
                color = CATEGORIES[cat]["color"]
                st.markdown(
                    f"""<div class="bin-card" style="background:{color}22;border-left:10px solid {color};">
                    <h2>{CATEGORIES[cat]['icon']} {cat}</h2>
                    <p><b>Put it in:</b> <span style="color:{color};font-weight:700">{info['target']}</span></p>
                    <p><b>Detected item type:</b> {info['display']} ({msp:.1%})</p></div>""",
                    unsafe_allow_html=True,
                )
                st.success("**How to dispose of it:**\n\n" + info["instruction"])

            st.session_state.history.append({"category": "Uncertain - asked user" if uncertain else cat})

            cat_probs = aggregate_to_categories(probs, CLASS_NAMES)[0]
            fig = go.Figure(go.Bar(
                x=cat_probs, y=CATEGORY_NAMES, orientation="h",
                marker_color=[CATEGORIES[c]["color"] for c in CATEGORY_NAMES],
                text=[f"{p:.1%}" for p in cat_probs], textposition="auto",
            ))
            fig.update_layout(title="Probability per cafeteria bin", height=260, margin=dict(l=10, r=10, t=40, b=10),
                              xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("All 12 fine-grained class probabilities"):
                df = pd.DataFrame({
                    "Item type": [CLASS_INFO[c]["display"] for c in CLASS_NAMES],
                    "Cafeteria category": [CLASS_INFO[c]["category"] for c in CLASS_NAMES],
                    "Probability": probs,
                }).sort_values("Probability", ascending=False)
                st.dataframe(df.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)

    if st.session_state.history:
        with st.expander(f"Session summary ({len(st.session_state.history)} scans - counts only, no images kept)"):
            st.bar_chart(pd.Series([h["category"] for h in st.session_state.history]).value_counts())
            if st.button("Clear session"):
                st.session_state.history = []
                st.rerun()

# ===========================================================================
# TAB 2 - DATASET & EVALUATION
# ===========================================================================
with tab_eval:
    if SPLIT is None or REPORT is None:
        st.warning("Run `python train.py` and `python evaluate.py` to generate the dataset split and evaluation.")
    else:
        st.subheader("Categories")
        cat_rows = []
        for c in CATEGORY_NAMES:
            subs = [CLASS_INFO[k]["display"] for k in CLASS_NAMES if CLASS_INFO[k]["category"] == c]
            cat_rows.append({"Cafeteria category": f"{CATEGORIES[c]['icon']} {c}", "Bin": CATEGORIES[c]["bin"],
                             "Dataset classes mapped to it": ", ".join(subs)})
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

        st.subheader("Dataset & documented split")
        st.markdown(
            f"**Source:** Garbage Classification - 12 classes (Mostafa Abla, Kaggle; mirrored on Hugging Face). "
            f"**{SPLIT['total_images']:,} images**, stratified **70 / 15 / 15** split with fixed seed "
            f"`{SPLIT['seed']}`. Every file and its split is listed in `artifacts/split.csv`. "
            f"The test split was used only once, by `evaluate.py`."
        )
        rows = []
        for c, v in SPLIT["per_class"].items():
            rows.append({"Class": CLASS_INFO[c]["display"], "Category": v["category"], "Train": v["train"],
                         "Val": v["val"], "Test": v["test"], "Total": v["total"]})
        split_df = pd.DataFrame(rows)
        tot = split_df[["Train", "Val", "Test", "Total"]].sum()
        split_df.loc[len(split_df)] = ["TOTAL", "", *tot.tolist()]
        cA, cB = st.columns([1.4, 1])
        cA.dataframe(split_df, use_container_width=True, hide_index=True)
        cat_counts = pd.DataFrame(rows).groupby("Category")[["Train", "Val", "Test"]].sum().reset_index()
        cB.plotly_chart(px.bar(cat_counts, x="Category", y=["Train", "Val", "Test"], title="Images per category",
                               height=380), use_container_width=True)
        st.caption("Preprocessing: RGB conversion, resize to 224×224, ImageNet normalisation. Training-only "
                   "augmentation: random resized crop, flip, rotation, ColorJitter (lighting) and RandomErasing "
                   "(occlusion). 'Clothes' is over-represented (~35%), so the loss uses class weights and we report "
                   "macro-averaged metrics.")

        split_csv = load_split()
        if split_csv is not None:
            with st.expander("Representative training images per class"):
                cols = st.columns(6)
                for i, c in enumerate(CLASS_NAMES):
                    p = split_csv[(split_csv["class"] == c) & (split_csv.split == "train")].sample(1, random_state=1)
                    cols[i % 6].image(str(ROOT / p.path.iloc[0]), caption=CLASS_INFO[c]["display"],
                                      use_container_width=True)

        st.subheader("Test-set results")
        fg, cg = REPORT["fine_grained"], REPORT["category"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("5-bin accuracy", f"{cg['accuracy']:.2%}")
        k2.metric("5-bin macro F1", f"{cg['macro_avg']['f1-score']:.2%}")
        k3.metric("12-class accuracy", f"{fg['accuracy']:.2%}")
        k4.metric("12-class macro F1", f"{fg['macro_avg']['f1-score']:.2%}")

        c1, c2 = st.columns(2)
        c1.image(str(ARTIFACTS / "confusion_matrix_5.png"), caption="Cafeteria categories", use_container_width=True)
        c2.image(str(ARTIFACTS / "confusion_matrix_12.png"), caption="Fine-grained classes", use_container_width=True)

        per = pd.DataFrame([
            {"Class": CLASS_INFO[c]["display"], "Precision": v["precision"], "Recall": v["recall"],
             "F1": v["f1-score"], "Test images": int(v["support"])}
            for c, v in fg["per_class"].items()
        ])
        per5 = pd.DataFrame([
            {"Category": c, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1-score"],
             "Test images": int(v["support"])} for c, v in cg["per_class"].items()
        ])
        fmt = {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
        c1, c2 = st.columns(2)
        c1.markdown("**Per category**")
        c1.dataframe(per5.style.format(fmt), use_container_width=True, hide_index=True)
        c2.markdown("**Per class**")
        c2.dataframe(per.style.format(fmt), use_container_width=True, hide_index=True)

        st.subheader("Uncertainty gate")
        g = REPORT["gate_in_distribution"]
        ood = REPORT.get("ood", {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test images answered", f"{g['coverage']:.1%}")
        k2.metric("Accuracy on answered", f"{g['accuracy_on_accepted']:.2%}")
        k3.metric("Mistakes caught by gate", f"{g['errors_caught_by_gate']} / {g['errors_total']}")
        if "imagenette" in ood:
            k4.metric("Non-waste photos flagged", f"{ood['imagenette']['flag_rate']:.1%}",
                      help=f"{ood['imagenette']['n']} Imagenette photos (dogs, churches, parachutes...)")
        st.markdown(
            f"Default gate: flag if confidence < **{g['conf_threshold']}** or normalized entropy > "
            f"**{g['entropy_threshold']}**. "
            + (f"On out-of-scope photos the AUROC of MSP is **{ood['imagenette']['auroc_msp']:.3f}** and of entropy "
               f"**{ood['imagenette']['auroc_entropy']:.3f}**. " if "imagenette" in ood else "")
            + (f"Synthetic patterns (noise / blank tray / checkerboard / gradients) flagged: "
               f"**{ood['synthetic_patterns']['flag_rate']:.0%}**." if "synthetic_patterns" in ood else "")
        )
        sw = pd.DataFrame(g["threshold_sweep"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sw.conf_threshold, y=sw.coverage, name="coverage (answered)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=sw.conf_threshold, y=sw.accuracy_on_accepted, name="accuracy on answered",
                                 mode="lines+markers"))
        fig.update_layout(title="Trade-off: confidence threshold vs coverage and accuracy (test split)",
                          xaxis_title="τ_conf", yaxis=dict(tickformat=".0%"), height=330)
        st.plotly_chart(fig, use_container_width=True)
        if "imagenette" in ood:
            st.caption("Top tentative guesses for out-of-scope photos (before the gate): " + ", ".join(
                f"{CLASS_INFO[k]['display']} ({v})" for k, v in ood["imagenette"]["most_common_wrong_class"].items()))

        st.subheader("Failure cases (most confident mistakes on the test set)")
        fails = REPORT["gallery"]["failures"]
        cols = st.columns(4)
        for i, f in enumerate(fails[:12]):
            tag = "🟠 flagged by gate" if f["flagged"] else ("🟢 same bin anyway" if f["same_bin"] else "🔴 wrong bin")
            cols[i % 4].image(str(ARTIFACTS / f["file"]), use_container_width=True,
                              caption=f"True: {CLASS_INFO[f['true']]['display']} → Pred: "
                                      f"{CLASS_INFO[f['pred']]['display']} ({f['confidence']:.0%}) · {tag}")
        with st.expander("Success cases (one per class)"):
            cols = st.columns(6)
            for i, s in enumerate(REPORT["gallery"]["successes"]):
                cols[i % 6].image(str(ARTIFACTS / s["file"]), use_container_width=True,
                                  caption=f"{CLASS_INFO[s['true']]['display']} ({s['confidence']:.0%})")

        log = load_json("training_log.json")
        if log:
            with st.expander("Training log"):
                st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 3 - LIGHTING & OCCLUSION
# ===========================================================================
with tab_light:
    st.subheader("How lighting and occlusion affect predictions")
    if REPORT is None:
        st.warning("Run `python evaluate.py` to measure robustness.")
    else:
        rb = REPORT["robustness"]
        n_sub = sum(1 for _ in CLASS_NAMES) * 40
        st.markdown(
            f"Measured on a stratified subset of the **test split** (~{n_sub} images, 40 per class) by applying the "
            "same degradations as the sidebar simulator. Lines show 12-class accuracy, 5-bin accuracy, the share "
            "of images the safety gate flags, and the accuracy on the images it still answers."
        )
        names = {"gamma": "Lighting gamma (<1 = dim, >1 = overexposed)", "occlusion_pct": "Central occlusion (% area)",
                 "glare": "Overhead glare strength", "blur": "Blur radius (px)"}
        cols = st.columns(2)
        for i, (param, rows) in enumerate(rb.items()):
            df = pd.DataFrame(rows)
            fig = go.Figure()
            for col, label, dash in [("accuracy_12", "12-class accuracy", None), ("accuracy_5", "5-bin accuracy", None),
                                     ("flag_rate", "flagged by gate", "dash"),
                                     ("accuracy_on_accepted", "accuracy on answered", "dot")]:
                fig.add_trace(go.Scatter(x=df["value"], y=df[col], name=label, mode="lines+markers",
                                         line=dict(dash=dash)))
            fig.update_layout(title=names[param], yaxis=dict(range=[0, 1.02], tickformat=".0%"), height=330,
                              margin=dict(t=50, b=30), legend=dict(orientation="h", y=-0.25))
            cols[i % 2].plotly_chart(fig, use_container_width=True)

        def at(param, value):
            return next(r for r in rb[param] if r["value"] == value)

        base = at("gamma", 1.0)
        dark, bright = at("gamma", 0.35), at("gamma", 2.2)
        occ40, occ60 = at("occlusion_pct", 40), at("occlusion_pct", 60)
        gl = at("glare", 0.8)
        st.markdown(
            f"""
**What we observed (measured, not assumed):**
- **Clean images:** {base['accuracy_12']:.1%} accuracy, {base['flag_rate']:.1%} flagged.
- **Dim lighting (γ=0.35):** accuracy {dark['accuracy_12']:.1%}, flagged {dark['flag_rate']:.1%}. Dark images lose
  colour and texture cues - glass colours and food textures are the first to suffer.
- **Overexposure (γ=2.2):** accuracy {bright['accuracy_12']:.1%}, flagged {bright['flag_rate']:.1%}. Washed-out
  whites make paper, plastic and clear glass look alike.
- **Glare (0.8):** accuracy {gl['accuracy_12']:.1%}, flagged {gl['flag_rate']:.1%}. Specular highlights hide shape
  and label details.
- **Occlusion 40% / 60%:** accuracy {occ40['accuracy_12']:.1%} / {occ60['accuracy_12']:.1%}, flagged
  {occ40['flag_rate']:.1%} / {occ60['flag_rate']:.1%}. Once the item's centre is hidden, the model relies on edges
  and background.
- **The gate helps:** as conditions get worse, more images are flagged, and accuracy on the images that *are*
  answered stays higher than raw accuracy - the assistant asks for help instead of guessing.

**Mitigations:** training-time ColorJitter + RandomErasing (implemented); the confidence/entropy gate with a
"retake photo / check label" prompt (implemented); a diffused LED ring light and a fixed camera position at the
bin (proposed hardware); multi-frame voting from the live camera (proposed).
"""
        )

    st.subheader("Try it on one image")
    samples = demo_samples()
    if samples:
        pick = st.selectbox("Sample", list(samples), key="light_pick")
        base_img = Image.open(samples[pick]).convert("RGB")
        variants = [
            ("Original", {}),
            ("Dim (γ=0.35)", {"gamma": 0.35}),
            ("Overexposed (γ=2.2)", {"gamma": 2.2}),
            ("Glare 0.8", {"glare": 0.8}),
            ("Occlusion 40%", {"occlusion_pct": 40}),
            ("Blur 6", {"blur": 6}),
        ]
        imgs = [apply_perturbations(base_img, **kw) for _, kw in variants]
        probs = predict_probs(model, imgs, DEVICE)
        cols = st.columns(len(variants))
        for col, (name, _), im, p in zip(cols, variants, imgs, probs):
            top = int(np.argmax(p))
            unc, _ = gate(float(p[top]), float(normalized_entropy(p)), conf_t, ent_t)
            col.image(im, use_container_width=True)
            col.markdown(f"**{name}**  \n{CLASS_INFO[CLASS_NAMES[top]]['display']} · {p[top]:.0%}  \n"
                         + ("🟠 asks for help" if unc else "🟢 confident"))

# ===========================================================================
# TAB 4 - ARCHITECTURE, RESPONSIBLE AI & SDGs
# ===========================================================================
with tab_arch:
    st.subheader("System architecture")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#F1F8E9", fontname="Helvetica"];
          input [label="Input\\nUpload / Camera\\n(in-memory only)", fillcolor="#E3F2FD"];
          pre [label="Pre-processing\\nEXIF fix, RGB,\\n224x224, normalise"];
          model [label="ConvNeXt-Tiny\\n(ImageNet pretrained,\\nfine-tuned on 12 classes)", fillcolor="#FFF8E1"];
          soft [label="Softmax\\nP(class | image)"];
          gate [label="Uncertainty gate\\nMSP >= τ_conf AND\\nH_norm <= τ_ent ?", shape=diamond, fillcolor="#FFF3E0"];
          map [label="12 → 5 bin mapper\\n+ disposal instruction", fillcolor="#E8F5E9"];
          help [label="Ask user:\\ncheck label / retake /\\nask staff → manual bin", fillcolor="#FFEBEE"];
          input -> pre -> model -> soft -> gate;
          gate -> map [label="confident"];
          gate -> help [label="uncertain /\\nout-of-scope"];
        }
        """
    )
    st.markdown(
        r"""
**Uncertainty gate.** For softmax output $p$ over $K=12$ classes:
$\text{MSP}=\max_i p_i$ and $H_{norm} = -\frac{1}{\ln K}\sum_i p_i \ln p_i \in [0,1]$.
An image is flagged if $\text{MSP} < \tau_{conf}$ **or** $H_{norm} > \tau_{ent}$ (defaults 0.65 / 0.45, adjustable
in the sidebar). MSP catches a single weak guess; entropy catches probability spread across many classes.

**Training pipeline** (`train.py`): stratified 70/15/15 split (seed 42) → ConvNeXt-Tiny with ImageNet weights →
AdamW + one-cycle LR, mixed precision, class-weighted cross-entropy → best checkpoint chosen on validation macro-F1.
**Evaluation** (`evaluate.py`): one pass on the untouched test split, plus out-of-scope photos (Imagenette) and
synthetic lighting / occlusion / glare / blur sweeps. Every number in this app is read from
`artifacts/evaluation_report.json`.
"""
    )

    st.subheader("Responsible AI")
    st.markdown(
        """
- **Privacy:** images are processed in memory and discarded; nothing is written to disk or sent to a cloud
  service. Only per-session bin counts are kept, and only in the browser session. Users are asked to photograph the
  item, not people.
- **Bias & domain shift:** the dataset is mostly single items on clean backgrounds, collected outside India.
  Local items (banana-leaf plates, foil-lined pouches, tetra packs, steel tumblers) and cluttered trays are not
  represented, so accuracy in a real VIT cafeteria will be lower than the test score. The gate reduces the damage
  by abstaining, and the "I checked the label" choice can be logged (with consent) to build a local dataset.
- **Category simplification:** glass, batteries, clothes and shoes are grouped as *Non-Recyclable / Special Waste*
  because the brief defines five categories; their instructions still route batteries to e-waste and glass to a
  separate crate.
- **Safety & human oversight:** hazardous items (batteries, broken glass) always get a "hand it to staff" instruction;
  low-confidence predictions are never shown as a bin decision.
- **Accessibility:** icons + text (not colour alone) for bins, plain-language instructions, large result card,
  works on a phone browser with the phone camera.
- **Security:** uploads are only decoded as images (Pillow); the model checkpoint is loaded with
  `torch.load(weights_only=True)`.
        """
    )

    st.subheader("UN Sustainable Development Goals")
    s1, s2, s3 = st.columns(3)
    s1.markdown("#### SDG 12.5\nResponsible consumption & production - less contamination means more paper, "
                "plastic and metal actually gets recycled.")
    s2.markdown("#### SDG 11.6\nSustainable cities & communities - better municipal / campus waste management "
                "and less manual sorting by sanitation staff.")
    s3.markdown("#### SDG 13\nClimate action - food scraps diverted to composting / biogas instead of landfill "
                "reduce methane emissions.")

    st.subheader("Implemented vs. future work")
    a, b = st.columns(2)
    a.markdown(
        """
**Implemented (working today)**
- Fine-tuned ConvNeXt-Tiny classifier, 12 classes → 5 cafeteria bins
- Upload + live camera interface with disposal instructions
- MSP + entropy uncertainty gate with "check label / ask staff" flow and manual override
- Documented 70/15/15 split, test metrics, confusion matrices, failure gallery
- Measured lighting / occlusion / glare / blur robustness + live simulator
        """
    )
    b.markdown(
        """
**Next steps (proposed)**
- Collect & label photos from the VIT cafeteria to fine-tune on local items
- Multi-item tray detection (YOLO + TACO dataset)
- Edge deployment (ONNX / INT8 on Raspberry Pi) at the bin with an LED ring light
- Temperature scaling / energy-based OOD score for better calibration
- Dashboard for the sustainability team (contamination rates per bin, peak hours)
        """
    )

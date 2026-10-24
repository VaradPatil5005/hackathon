"""
build_deliverables.py
Builds the submission deliverables from the real evaluation results, in the Binzy / Ghumzy palette:
    deliverables/architecture.png, deliverables/chart_datasets.png, deliverables/chart_robustness.png
    deliverables/Smart_Waste_Segregation_Assistant.pptx   (7 slides, VIT suggested flow)
    deliverables/REPORT.md                                 (project report)
Run after train.py, evaluate.py, build_hybrid.py and generate_demo_samples.py.
Screenshots of the app (deliverables/screens/*.jpg) are added to the PPT when present.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from waste_core import ARTIFACTS, CATEGORIES, CATEGORY_NAMES, CLASS_INFO, ROOT  # noqa: E402

OUT = ROOT / "deliverables"
SCREENS = OUT / "screens"
FONT = "Segoe UI"
C = {"brand900": "#2f4a10", "brand700": "#4e7a1a", "brand500": "#6c9a22", "brand300": "#8dc63f", "brand100": "#e6f0c8",
     "paper": "#f2f2ec", "ink": "#1a1a17", "muted": "#6e6e66", "saffron": "#e2791b", "line": "#e4e4dc", "grey": "#b9b9ae"}
DS_NAME = {"kaggle_test": "Kaggle test split", "trashnet": "TrashNet (external)", "external2": "External set #2"}


def rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def pct(x, d=1):
    return "-" if x is None else f"{x * 100:.{d}f}%"


plt.rcParams.update({"font.family": "Segoe UI", "axes.edgecolor": C["line"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.spines.top": False,
                     "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec=None, fs=10.5, color=None, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.18", fc=fc, ec=ec or fc, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or C["ink"],
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color or C["muted"]))


def draw_architecture(path, cfg):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.patch.set_facecolor(C["paper"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, 0.1, 2.0, 1.8, 1.3, "Photo\nupload / camera /\nlive scan\n(in memory only)", "#ffffff", C["line"], 9.5)
    box(ax, 2.5, 3.75, 2.9, 1.05, "SigLIP-so400m zero-shot\n202 described waste items\n+ 52 'not waste' concepts", C["brand100"], fs=9.3)
    box(ax, 2.5, 2.1, 2.9, 1.05, "SigLIP embedding\n+ trained linear head\n(12 classes)", C["brand100"], fs=9.3)
    box(ax, 2.5, 0.45, 2.9, 1.05, "ConvNeXt-Tiny\nfine-tuned on 70% split\n(12 classes)", C["brand100"], fs=9.3)
    for y in (4.27, 2.62, 0.97):
        arrow(ax, 1.95, 2.65, 2.45, y)
    box(ax, 6.0, 2.0, 2.2, 1.3, f"Weighted fusion\nw = {cfg['w_zs']:.1f} / {cfg['w_lp']:.1f} / {cfg['w_cn']:.1f}\n12 classes -> 5 bins",
        "#ffffff", C["brand500"], 9.5)
    for y in (4.27, 2.62, 0.97):
        arrow(ax, 5.45, y, 5.95, 2.65)
    box(ax, 8.8, 2.0, 2.3, 1.3, f"Safety gate\nnot-waste > {cfg['ood_threshold']:.2f}?\nconf < {cfg['conf_threshold']:.2f} or\nentropy > {cfg['entropy_threshold']:.2f}?",
        "#fff3e0", C["saffron"], 9.2)
    arrow(ax, 8.25, 2.65, 8.75, 2.65)
    box(ax, 11.7, 3.7, 2.2, 1.2, "Bin + item name\n+ disposal steps\n+ item tip", C["brand700"], fs=9.8, color="white", bold=True)
    box(ax, 11.7, 2.05, 2.2, 1.2, "Not sure:\ncheck label / retake /\nask staff / pick bin", "#fde7c7", fs=9.5)
    box(ax, 11.7, 0.4, 2.2, 1.2, "Out of scope:\n'this isn't waste'", "#dcecf2", fs=9.5)
    arrow(ax, 11.15, 2.9, 11.65, 4.3, C["brand700"])
    arrow(ax, 11.15, 2.65, 11.65, 2.65, C["saffron"])
    arrow(ax, 11.15, 2.4, 11.65, 1.0, "#2f7f9e")
    ax.text(0.1, 0.05, "Offline: stratified 70/15/15 split · ConvNeXt fine-tuning (ColorJitter + RandomErasing) · linear head on "
                       "SigLIP features · fusion weights tuned on validation + TrashNet · thresholds from in-scope data only",
            fontsize=9, color=C["muted"])
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=C["paper"])
    plt.close(fig)


def chart_datasets(hr, path):
    sets = list(hr["sets"].keys())
    names = [("convnext_only", "ConvNeXt only", C["grey"]), ("zero_shot_only", "SigLIP zero-shot", C["brand300"]),
             ("probe_only", "SigLIP + head", C["brand500"]), ("hybrid", "Hybrid (ours)", C["brand900"])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white")
    w = 0.2
    for i, (k, label, color) in enumerate(names):
        vals = [hr["sets"][s][k]["accuracy"] for s in sets]
        bars = ax.bar([j + (i - 1.5) * w for j in range(len(sets))], vals, w * 0.92, label=label, color=color)
        if k == "hybrid":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v * 100:.1f}%", ha="center", fontsize=9,
                        color=C["brand900"], fontweight="bold")
    ax.set_xticks(range(len(sets)), [DS_NAME.get(s, s) for s in sets], fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Bin accuracy (5 categories)")
    ax.grid(axis="y", color=C["line"])
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def chart_robustness(hr, path):
    titles = {"gamma": "Lighting γ", "occlusion_pct": "Occlusion %", "glare": "Glare", "blur": "Blur px"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (k, rows) in zip(axes, hr["robustness"].items()):
        x = [r["value"] for r in rows]
        ax.plot(x, [r["hybrid_accuracy"] for r in rows], "o-", color=C["brand900"], lw=2.2, label="Hybrid")
        ax.plot(x, [r["convnext_accuracy"] for r in rows], "s-", color=C["grey"], lw=2, label="ConvNeXt only")
        ax.plot(x, [r["hybrid_answered"] for r in rows], "^--", color=C["saffron"], lw=1.8, label="Hybrid answered")
        ax.set_title(titles[k], fontsize=11, color=C["ink"])
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(color=C["line"])
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PPT helpers
# ---------------------------------------------------------------------------
def bg(slide, color=C["paper"]):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)


def rect(slide, x, y, w, h, color, radius=True):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(color)
    s.line.fill.background()
    s.shadow.inherit = False
    if radius:
        s.adjustments[0] = 0.08
    return s


def text(slide, x, y, w, h, lines, size=16, color=C["ink"], bold=False, font=FONT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bullet = line.startswith("• ") or line.startswith("– ")
        p.text = line
        p.font.size = Pt(size - (2 if line.startswith("– ") else 0))
        p.font.bold = bold and not bullet
        p.font.name = font
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(6)
    return tb


def header(slide, kicker, title, n):
    bg(slide)
    band = rect(slide, 0, 0, 13.333, 1.45, C["brand900"], radius=False)
    band.fill.gradient()
    band.fill.gradient_angle = 160
    band.fill.gradient_stops[0].color.rgb = rgb(C["brand900"])
    band.fill.gradient_stops[1].color.rgb = rgb(C["brand500"])
    text(slide, 0.55, 0.18, 10, 0.4, kicker.upper(), 12, "#dfeccc", True)
    text(slide, 0.55, 0.5, 12, 0.8, title, 28, "#ffffff", True)
    text(slide, 10.9, 0.22, 2.2, 0.4, "Binzy", 18, "#ffffff", True)
    text(slide, 0.55, 7.05, 12, 0.35, f"VIT AI Innovate-A-Thon · Q6 Smart Waste Segregation Assistant · {n}/7", 10, C["muted"])


def card(slide, x, y, w, h, title=None, lines=None, size=14, title_color=C["brand900"]):
    rect(slide, x, y, w, h, "#ffffff")
    yy = y + 0.15
    if title:
        text(slide, x + 0.2, yy, w - 0.4, 0.45, title, size + 2, title_color, True)
        yy += 0.5
    if lines:
        text(slide, x + 0.2, yy, w - 0.4, h - (yy - y) - 0.1, lines, size)


def picture(slide, path, x, y, w=None, h=None):
    if path.exists():
        kw = {}
        if w:
            kw["width"] = Inches(w)
        if h:
            kw["height"] = Inches(h)
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def table(slide, rows, x, y, w, col_w, size=12):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.34 * len(rows)))
    t = shape.table
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(C["brand900"] if r == 0 else ("#ffffff" if r % 2 else "#f7f7f2"))
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            para.font.name = FONT
            para.font.bold = r == 0 or c == len(row) - 1
            para.font.color.rgb = rgb("#ffffff" if r == 0 else C["ink"])
    for c, cw in enumerate(col_w):
        t.columns[c].width = Inches(cw)


def build_pptx(cr, hr, split, path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    cfg = hr["config"]
    kt = hr["sets"]["kaggle_test"]["hybrid"]
    ext_key = "external2" if "external2" in hr["sets"] else "trashnet"
    ext = hr["sets"][ext_key]

    # 1 - Problem & users
    s = prs.slides.add_slide(blank)
    header(s, "Problem & intended users", "One wrong item contaminates the whole bin", 1)
    card(s, 0.55, 1.75, 7.2, 5.1, "The problem", [
        "• A campus cafeteria produces mixed waste: paper, plastic, metal, food scraps and non-recyclables",
        "• Students decide in ~2 seconds and often guess - wrappers, paper cups, tetra packs and leftovers are confusing",
        "• Wrong-bin disposal contaminates recycling and forces sanitation staff to re-sort by hand",
        "• Goal: photo of any item → the right bin + how to dispose of it, and an honest 'not sure' when needed",
        "Intended users",
        "• Students & faculty at the bin (phone or kiosk camera)",
        "• Cafeteria / housekeeping staff and the campus sustainability team",
    ], 15)
    for i, name in enumerate(["50_Real_chocolate_wrapper.jpg", "51_Real_leftover_cake.jpg", "52_Real_water_bottle.jpg", "07_Metal.jpg"]):
        picture(s, ROOT / "demo_samples" / name, 8.1 + (i % 2) * 2.55, 1.8 + (i // 2) * 2.55, w=2.35, h=2.35)
    # 2 - Idea & innovation
    s = prs.slides.add_slide(blank)
    header(s, "Proposed idea & innovation", "A waste sorter that understands any item - and knows when it doesn't", 2)
    items = [
        ("Open vocabulary", f"Google SigLIP compares the photo with {hr['vocabulary']['waste_items']} written descriptions "
                            "(chocolate wrapper, leftover cake, clay kulhad, tetra pack…) - recognises items never in any training set."),
        ("Three experts, one vote", f"SigLIP zero-shot + SigLIP with a trained head + fine-tuned ConvNeXt, fused with weights "
                                    f"{cfg['w_zs']:.1f}/{cfg['w_lp']:.1f}/{cfg['w_cn']:.1f} tuned on validation data."),
        ("Honest safety net", "'Not waste' concepts reject selfies, phones, scenery. Low confidence or high entropy → "
                              "'check the label, retake, ask staff', plus a manual bin picker."),
        ("Item-specific advice", "Bin + target + steps + item tips: multi-layer wrappers, greasy pizza boxes, batteries to e-waste."),
        ("Fast & private", "~90 ms per photo on a laptop GPU, live auto-scan from the camera, nothing stored or sent to a cloud."),
        ("Stress-tested", "Live lighting / glare / occlusion / blur simulator and measured robustness curves."),
    ]
    for i, (t, d) in enumerate(items):
        card(s, 0.55 + (i % 3) * 4.12, 1.75 + (i // 3) * 2.6, 3.95, 2.4, t, [d], 13)
    # 3 - Architecture
    s = prs.slides.add_slide(blank)
    header(s, "System architecture & technical approach", "Three experts → fusion → safety gate", 3)
    picture(s, OUT / "architecture.png", 0.45, 1.6, w=12.4)
    text(s, 0.55, 6.25, 12.3, 0.8, [
        f"Data: {split['total_images']:,} images, 12 classes, stratified 70/15/15 ({split['split_totals']['train']:,} / "
        f"{split['split_totals']['val']:,} / {split['split_totals']['test']:,}), seed 42 · Stack: PyTorch, Hugging Face "
        "Transformers (SigLIP), torchvision (ConvNeXt), scikit-learn, FastAPI, vanilla-JS web app"], 12, C["muted"])
    # 4 - Demo & workflow
    s = prs.slides.add_slide(blank)
    header(s, "Prototype demonstration & user workflow", "Snap → bin → done (or 'not sure, please check')", 4)
    card(s, 0.55, 1.75, 4.6, 5.1, "Live demo", [
        "• run_app.bat → http://localhost:8000",
        "• Scan: camera, upload, drag-drop, paste, live auto-scan",
        "• Result: bin, target, item name, steps, tip, 'why this answer'",
        "• Insights: split, metrics, external datasets, failures",
        "• Stress test: lighting / glare / occlusion / blur sliders",
        "Script",
        "• Water bottle → Yellow bin · leftover cake → Green bin",
        "• Chocolate wrapper → Red/Black (multi-layer tip)",
        "• Battery → e-waste box",
        "• Dog / noise photo → 'this isn't waste'",
        "• Occlusion slider to 60% → 'not sure - please check'",
    ], 12)
    shots = [p for p in [SCREENS / "app_result.jpg", SCREENS / "app_uncertain.jpg", SCREENS / "app_home.jpg"] if p.exists()]
    for i, p in enumerate(shots[:2]):
        picture(s, p, 5.4 + i * 3.95, 1.75, w=3.8)
    # 5 - Testing & evaluation
    s = prs.slides.add_slide(blank)
    header(s, "Testing, evaluation results & limitations", "Measured on images the models never trained on", 5)
    rows = [["Dataset", "n", "ConvNeXt only", "Hybrid"]]
    for k, v in hr["sets"].items():
        rows.append([DS_NAME.get(k, k) + ("" if v["held_out"] else " (tuning)"), v["hybrid"]["n"],
                     pct(v["convnext_only"]["accuracy"]), pct(v["hybrid"]["accuracy"])])
    rows.append(["Non-waste photos refused", hr["ood"]["imagenette"]["n"], pct(cr["ood"]["imagenette"]["flag_rate"]),
                 pct(hr["ood"]["imagenette"]["not_answered"])])
    table(s, rows, 0.55, 1.75, 6.3, [2.9, 0.8, 1.3, 1.3], 12)
    text(s, 0.55, 1.95 + 0.34 * len(rows), 6.3, 2.6, [
        f"• Kaggle test: macro-F1 {pct(kt['macro_f1'])}, answers {pct(kt['answered'])} of photos with {pct(kt['accuracy_on_answered'])} accuracy",
        f"• {DS_NAME.get(ext_key)}: hybrid {pct(ext['hybrid']['accuracy'])} vs ConvNeXt {pct(ext['convnext_only']['accuracy'])}",
        "• Limits: one item per photo; occlusion > 40% hurts most; cluttered trays and local items not in training data",
    ], 12)
    picture(s, OUT / "chart_datasets.png", 7.05, 1.7, w=5.8)
    picture(s, OUT / "chart_robustness.png", 7.05, 4.85, w=5.8)
    # 6 - Responsible AI & SDGs
    s = prs.slides.add_slide(blank)
    header(s, "Responsible AI & SDG alignment", "Private, honest, and built for a greener campus", 6)
    rai = [("Privacy", "Photos processed in memory on the local machine, never saved or uploaded; session stats are counts only."),
           ("Bias & domain shift", "Training photos are clean single items from outside India; the open-vocabulary expert covers local items, "
                                   "but real cafeteria accuracy will be lower than test scores."),
           ("Human oversight", "Uncertain / out-of-scope photos never get a bin decision; users check the label, ask staff or pick the bin."),
           ("Safety & accessibility", "Batteries and broken glass always go to staff. Icons + text, plain language, phone-friendly, dark mode.")]
    for i, (t, d) in enumerate(rai):
        card(s, 0.55 + (i % 2) * 3.55, 1.75 + (i // 2) * 2.6, 3.4, 2.4, t, [d], 12)
    for i, (n, t, d, col) in enumerate([("12.5", "Responsible consumption", "Cleaner streams → more material actually recycled", "#bf8b2e"),
                                         ("11.6", "Sustainable cities", "Better campus waste management, less manual sorting", "#fd9d24"),
                                         ("13", "Climate action", "Food scraps to compost/biogas, not landfill → less methane", "#3f7e44")]):
        y = 1.75 + i * 1.73
        rect(s, 7.85, y, 4.95, 1.55, "#ffffff")
        rect(s, 8.0, y + 0.2, 1.15, 1.15, col)
        text(s, 8.0, y + 0.47, 1.15, 0.6, f"SDG {n}", 15, "#ffffff", True)
        text(s, 9.3, y + 0.15, 3.4, 0.5, t, 15, C["brand900"], True)
        text(s, 9.3, y + 0.6, 3.4, 0.9, d, 12)
    # 7 - Conclusion
    s = prs.slides.add_slide(blank)
    header(s, "Conclusion", "Implemented today · what comes next", 7)
    card(s, 0.55, 1.75, 6.0, 5.1, "Implemented (working now)", [
        f"• Hybrid recogniser - {pct(kt['accuracy'])} bin accuracy on {kt['n']:,} held-out test photos, "
        f"{pct(ext['hybrid']['accuracy'])} on an unseen external dataset",
        "• Open vocabulary: 202 everyday waste items incl. Indian cafeteria waste",
        "• Camera, upload and live auto-scan web app (Ghumzy design language)",
        "• 5 bins, item tips, 'not waste' + 'not sure' safety net with manual override",
        "• Documented split, external-dataset tests, failure gallery, robustness study",
    ], 14)
    card(s, 6.8, 1.75, 6.0, 5.1, "Next steps", [
        "• Collect & label photos in the VIT cafeteria; fine-tune on them",
        "• Multi-item tray detection (YOLO + TACO dataset)",
        "• Edge box at the bin: ONNX / INT8, LED ring light, lid that opens the right bin",
        "• Contamination dashboard for the sustainability team",
        "• Hindi / Tamil labels and voice prompts",
    ], 14, C["saffron"])
    prs.save(path)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def build_report(cr, hr, split, log, path):
    fg, cg = cr["fine_grained"], cr["category"]
    cfg = hr["config"]
    L = ["# Binzy - Smart Waste Segregation Assistant: Project Report", "",
         "**Event:** VIT AI Innovate-A-Thon, Question 6  ",
         "**Deliverable:** working web app (camera, upload, live scan) backed by a hybrid computer-vision recogniser", "",
         "## 1. Problem and intended users",
         "A campus cafeteria generates mixed waste (paper, plastic, metal, food scraps, non-recyclables). Items in the wrong "
         "bin contaminate recycling and increase manual sorting. Binzy classifies a photo of one waste item into one of five "
         "disposal categories, names the item, shows the target bin and disposal steps, and - when it is unsure or the image "
         "is not waste - asks the user to check the label, retake the photo or ask staff.", "",
         "Users: students/faculty at the bin (phone or kiosk camera); cafeteria staff and the sustainability team.", "",
         "## 2. Categories", "| Category | Bin | Dataset classes |", "|---|---|---|"]
    for c in CATEGORY_NAMES:
        subs = ", ".join(CLASS_INFO[k]["display"] for k in split["class_names"] if CLASS_INFO[k]["category"] == c)
        L.append(f"| {c} | {CATEGORIES[c]['bin']} | {subs} |")
    L += ["", "Glass, batteries, textiles and shoes are grouped as Non-Recyclable / Special Waste because the brief defines "
          "five categories, but each gets its own instruction (glass crate, e-waste box, donation).", "",
          "## 3. Data and documented split",
          f"Training data: Garbage Classification, 12 classes (Mostafa Abla, Kaggle; Hugging Face mirror "
          f"`UdaraChamidu/Garbage-Classification-with-12-classes`), {split['total_images']:,} images, stratified 70/15/15 "
          f"split with seed {split['seed']} (`train.py`, file list in `artifacts/split.csv`).", "",
          "| Class | Category | Train | Val | Test | Total |", "|---|---|---|---|---|---|"]
    for c, v in split["per_class"].items():
        L.append(f"| {CLASS_INFO[c]['display']} | {v['category']} | {v['train']} | {v['val']} | {v['test']} | {v['total']} |")
    t = split["split_totals"]
    L += [f"| **Total** | | **{t['train']}** | **{t['val']}** | **{t['test']}** | **{split['total_images']}** |", "",
          "External data (never used for training):",
          "- **TrashNet** (Stanford, 2,527 photos, 6 classes) - used to tune fusion weights, reported as a tuning set.",
          "- **External set #2** (`dmedhi/garbage-image-classification-detection`, validation split, 397 photos) - held out.",
          "- **Imagenette** validation (1,000 everyday non-waste photos) - out-of-scope test.",
          "- **Synthetic patterns** (noise, blank tray, checkerboard, gradients) - out-of-scope test.", "",
          "## 4. Method",
          "Three experts produce probabilities over the 12 classes:",
          f"1. **SigLIP-so400m zero-shot** (Google, `google/siglip-so400m-patch14-384`): image vs. text embeddings of "
          f"{hr['vocabulary']['waste_items']} vocabulary entries (≈200 described waste items, each mapped to a class, plus ≈50 "
          "'not waste' concepts), three prompt templates each. Gives open-vocabulary recognition and an explicit "
          "'not waste' probability.",
          f"2. **SigLIP linear head**: logistic regression (C={hr['probe_C']}) on SigLIP image embeddings of the training split "
          f"(validation accuracy {pct(hr['probe_val_acc'], 2)}).",
          f"3. **ConvNeXt-Tiny**: ImageNet-pretrained, fully fine-tuned for {cr['model'].get('epochs')} epochs (AdamW, one-cycle "
          f"LR, mixed precision, class-weighted loss, ColorJitter + RandomErasing), best epoch {cr['model'].get('best_epoch')} by "
          "validation macro-F1.", "",
          f"**Fusion:** p = {cfg['w_zs']:.1f}·zero-shot + {cfg['w_lp']:.1f}·head + {cfg['w_cn']:.1f}·ConvNeXt (temperature "
          f"{cfg['temperature']}), chosen by grid search to maximise mean bin accuracy on Kaggle validation + TrashNet. "
          "Class probabilities are summed into the 5 bins.", "",
          f"**Safety gate:** out of scope if the zero-shot 'not waste' mass > {cfg['ood_threshold']:.2f}; 'not sure' if the top "
          f"bin probability < {cfg['conf_threshold']:.2f} or the normalised entropy over the 5 bins > "
          f"{cfg['entropy_threshold']:.2f}. Thresholds are the 97th / 3rd percentiles on in-scope tuning photos - no "
          "out-of-scope data was used to set them.", "",
          "ConvNeXt training log:", "", "| Epoch | Train loss | Val accuracy | Val macro-F1 |", "|---|---|---|---|"]
    for e in log:
        L.append(f"| {e['epoch']} | {e['train_loss']:.4f} | {e['val_accuracy']:.4f} | {e['val_macro_f1']:.4f} |")
    L += ["", "## 5. Results", "### Bin accuracy on photos the models never trained on", "",
          "| Dataset | n | ConvNeXt only | SigLIP zero-shot | SigLIP head | **Hybrid** | Hybrid macro-F1 | Answered | Acc. when answered |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in hr["sets"].items():
        L.append(f"| {DS_NAME.get(k, k)}{'' if v['held_out'] else ' (tuning)'} | {v['hybrid']['n']} | "
                 f"{pct(v['convnext_only']['accuracy'])} | {pct(v['zero_shot_only']['accuracy'])} | {pct(v['probe_only']['accuracy'])} | "
                 f"**{pct(v['hybrid']['accuracy'])}** | {pct(v['hybrid']['macro_f1'])} | {pct(v['hybrid']['answered'])} | "
                 f"{pct(v['hybrid']['accuracy_on_answered'])} |")
    o = hr["ood"]
    L += ["", f"Out-of-scope: {pct(o['imagenette']['not_answered'])} of 1,000 Imagenette photos were refused "
          f"({pct(o['imagenette']['out_of_scope'])} as 'not waste', the rest as 'not sure'); synthetic patterns: "
          f"{pct(o['synthetic']['not_answered'])} refused. ConvNeXt alone with the old MSP/entropy gate flagged "
          f"{pct(cr['ood']['imagenette']['flag_rate'])} of Imagenette.", "",
          "![Accuracy by dataset](chart_datasets.png)", "",
          f"### ConvNeXt detail on the Kaggle test split ({cr['test_size']:,} images)",
          "| Metric | 5 bins | 12 classes |", "|---|---|---|",
          f"| Accuracy | {pct(cg['accuracy'], 2)} | {pct(fg['accuracy'], 2)} |",
          f"| Macro F1 | {pct(cg['macro_avg']['f1-score'], 2)} | {pct(fg['macro_avg']['f1-score'], 2)} |", "",
          "| Class | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
    for c, v in fg["per_class"].items():
        L.append(f"| {CLASS_INFO[c]['display']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {int(v['support'])} |")
    L += ["", "![5-bin confusion matrix](../artifacts/confusion_matrix_5.png)", "",
          "### Failure cases (most confident ConvNeXt test mistakes)", "| True | Predicted | Confidence | Same bin |", "|---|---|---|---|"]
    for f in cr["gallery"]["failures"][:10]:
        L.append(f"| {CLASS_INFO[f['true']]['display']} | {CLASS_INFO[f['pred']]['display']} | {pct(f['confidence'])} | "
                 f"{'yes' if f['same_bin'] else 'no'} |")
    L += ["", "Typical confusions: clear glass vs. plastic (transparent, similar shapes), paper vs. cardboard (same bin), "
          "trash vs. plastic packaging. Real-world photos from the team (chocolate wrapper, leftover cake) were misjudged by "
          "ConvNeXt alone and corrected by the open-vocabulary experts.", "",
          "## 6. Lighting and occlusion", "Hybrid vs. ConvNeXt on 240 test images (20 per class) with the app's perturbations:", ""]
    names = {"gamma": "Lighting gamma", "occlusion_pct": "Occlusion %", "glare": "Glare", "blur": "Blur radius"}
    for k, rows in hr["robustness"].items():
        L += [f"**{names[k]}**", "", "| Value | Hybrid acc. | ConvNeXt acc. | Hybrid answered | Hybrid acc. when answered |",
              "|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['value']} | {pct(r['hybrid_accuracy'])} | {pct(r['convnext_accuracy'])} | {pct(r['hybrid_answered'])} | "
                     f"{pct(r['hybrid_accuracy_on_answered'])} |")
        L.append("")
    L += ["![Robustness](chart_robustness.png)", "",
          "Discussion: dim light removes colour and texture cues; overexposure and glare wash out white paper, clear plastic and "
          "glass so they look alike; central occlusion hides the object so the models fall back on edges and background. "
          "Lighting is handled well thanks to ColorJitter training and SigLIP's broad pre-training; heavy occlusion (>40%) is "
          "the main weakness. As images degrade the gate answers fewer of them, so accuracy on answered photos stays higher "
          "than raw accuracy. Mitigations: augmentation (done), the gate with retake / check-label prompts (done), a diffused "
          "LED ring light and fixed camera at the bin (proposed), multi-frame voting in live mode (proposed).", "",
          "## 7. Responsible AI",
          "- **Privacy:** photos are processed in memory on the local machine and discarded; nothing is written to disk or sent "
          "to a cloud service. Session statistics are counts only.",
          "- **Bias / domain shift:** training photos are mostly clean single items from outside India. The open-vocabulary "
          "list adds Indian cafeteria items, but accuracy in a real cafeteria will be lower than the test scores.",
          "- **Human oversight:** uncertain or out-of-scope photos never receive a bin decision; users check the label, ask "
          "staff or choose the bin. Hazardous items (batteries, broken glass) always go to staff.",
          "- **Accessibility:** icons plus text, plain-language steps, large touch targets, works in a phone browser, dark mode.",
          "- **Security:** uploads limited to 15 MB, decoded only as images (decompression-bomb guard), model checkpoint loaded "
          "with `weights_only=True`, server binds to localhost unless `--lan` is passed.", "",
          "## 8. SDG alignment",
          "SDG 12.5 (substantially reduce waste through recycling), SDG 11.6 (municipal / campus waste management), SDG 13 "
          "(food waste to compost/biogas instead of landfill reduces methane).", "",
          "## 9. Implemented vs. future work",
          "**Implemented:** hybrid recogniser, open vocabulary, camera / upload / live scan web app, item-specific instructions, "
          "out-of-scope and uncertainty gate with manual override, documented split, external-dataset evaluation, robustness "
          "study and live stress test.", "",
          "**Future:** local VIT cafeteria dataset and fine-tuning, multi-item tray detection (YOLO + TACO), edge deployment at "
          "the bin, contamination dashboard, regional-language labels and voice prompts.", "",
          "## 10. How to run", "```", "run_app.bat               (opens http://localhost:8000)",
          "run_app.bat --lan         (also reachable from a phone on the same Wi-Fi)", "```",
          "Reproduce: `train.py` → `evaluate.py` → `build_hybrid.py` → `generate_demo_samples.py` → `build_deliverables.py`."]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    rd = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    cr, hr = rd(ARTIFACTS / "evaluation_report.json"), rd(ARTIFACTS / "hybrid_report.json")
    split, log = rd(ARTIFACTS / "split_summary.json"), rd(ARTIFACTS / "training_log.json")
    draw_architecture(OUT / "architecture.png", hr["config"])
    chart_datasets(hr, OUT / "chart_datasets.png")
    chart_robustness(hr, OUT / "chart_robustness.png")
    build_pptx(cr, hr, split, OUT / "Smart_Waste_Segregation_Assistant.pptx")
    build_report(cr, hr, split, log, OUT / "REPORT.md")
    print("Wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()

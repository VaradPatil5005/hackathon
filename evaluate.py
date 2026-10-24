"""
evaluate.py
Evaluates the trained checkpoint ONCE on the held-out test split and writes
every number the app shows. Nothing in the app's evaluation tab is hard-coded.

Usage:
    python evaluate.py [--ood_dir data/imagenette2-160/val] [--robust_per_class 40]
Outputs (in artifacts/):
    evaluation_report.json      all metrics (12-class, 5-category, gate, OOD, robustness)
    confusion_matrix_12.png     fine-grained confusion matrix (test)
    confusion_matrix_5.png      cafeteria-category confusion matrix (test)
    robustness_curves.png       lighting / occlusion / glare / blur sensitivity
    failures/*.jpg              misclassified test images (failure gallery)
    successes/*.jpg             correctly classified test images
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from waste_core import (  # noqa: E402
    ARTIFACTS,
    CATEGORY_NAMES,
    CLASS_INFO,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_ENTROPY_THRESHOLD,
    ROOT,
    apply_perturbations,
    eval_transform,
    load_trained_model,
    normalized_entropy,
)

SEED = 42


class PerturbedDataset(Dataset):
    def __init__(self, paths, perturb=None):
        self.paths, self.perturb, self.tf = list(paths), perturb or {}, eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        if self.perturb:
            img = apply_perturbations(img, **self.perturb)
        return self.tf(img)


def synthetic_ood_images(n_each: int = 20):
    """Blank trays, noise, checkerboards and colour gradients: clearly not waste items."""
    rng = np.random.default_rng(SEED)
    imgs = []
    for _ in range(n_each):
        imgs.append(Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)))
        g = int(rng.integers(60, 200))
        imgs.append(Image.fromarray(np.full((224, 224, 3), g, np.uint8)))
        s = int(rng.choice([14, 28, 56]))
        cb = ((np.indices((224, 224)) // s).sum(0) % 2 * 255).astype(np.uint8)
        imgs.append(Image.fromarray(np.stack([cb] * 3, -1)))
        c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, 224)[None, :, None]
        imgs.append(Image.fromarray((c1 * (1 - t) + c2 * t).repeat(224, 0).astype(np.uint8)))
    return imgs


@torch.no_grad()
def predict(model, device, paths=None, images=None, perturb=None, batch_size=64, workers=2):
    if images is not None:
        tf = eval_transform()
        batches = [torch.stack([tf(im.convert("RGB")) for im in images[i:i + batch_size]])
                   for i in range(0, len(images), batch_size)]
    else:
        batches = DataLoader(PerturbedDataset(paths, perturb), batch_size=batch_size,
                             num_workers=workers, pin_memory=True)
    out = []
    for x in batches:
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(x.to(device))
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def gate_mask(probs, tc=DEFAULT_CONF_THRESHOLD, te=DEFAULT_ENTROPY_THRESHOLD):
    """True where the image would be flagged as uncertain / out-of-scope."""
    return (probs.max(1) < tc) | (normalized_entropy(probs) > te)


def to_category(idx, class_names):
    return np.array([CATEGORY_NAMES.index(CLASS_INFO[class_names[i]]["category"]) for i in idx])


def plot_cm(cm, labels, title, path, figsize):
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "row-normalised recall"}, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_thumb(src, dst):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((320, 320))
        im.save(dst, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood_dir", default="data/imagenette2-160/val")
    ap.add_argument("--ood_samples", type=int, default=1000)
    ap.add_argument("--robust_per_class", type=int, default=40)
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    model, class_names, meta, device = load_trained_model()
    split = pd.read_csv(ARTIFACTS / "split.csv")
    test = split[split.split == "test"].reset_index(drop=True)
    val = split[split.split == "val"].reset_index(drop=True)
    test_paths = [ROOT / p for p in test.path]
    y_true = test.label.to_numpy()
    print(f"Evaluating on {len(test)} test images ({device})")

    # ---------------- 1. Clean test-set metrics ----------------
    probs = predict(model, device, paths=test_paths)
    np.savez_compressed(ARTIFACTS / "test_predictions.npz", probs=probs, labels=y_true)
    y_pred = probs.argmax(1)
    rep12 = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    cm12 = confusion_matrix(y_true, y_pred)
    yc_true, yc_pred = to_category(y_true, class_names), to_category(y_pred, class_names)
    rep5 = classification_report(yc_true, yc_pred, labels=range(5), target_names=CATEGORY_NAMES,
                                 output_dict=True, digits=4)
    cm5 = confusion_matrix(yc_true, yc_pred, labels=range(5))
    plot_cm(cm12, [CLASS_INFO[c]["display"] for c in class_names],
            f"12-class confusion matrix - test split (n={len(test)})", ARTIFACTS / "confusion_matrix_12.png", (10, 8))
    plot_cm(cm5, ["Paper & Card", "Plastic", "Metal", "Food/Organic", "Non-Recyc./Special"],
            f"5 cafeteria categories - test split (n={len(test)})", ARTIFACTS / "confusion_matrix_5.png", (7, 5.5))
    print(f"Test accuracy 12-class: {rep12['accuracy']:.4f} | 5-category: {rep5['accuracy']:.4f}")

    # ---------------- 2. Uncertainty gate on the in-distribution test set ----------------
    flagged = gate_mask(probs)
    correct = y_pred == y_true
    msp, ent = probs.max(1), normalized_entropy(probs)
    gate_in = {
        "conf_threshold": DEFAULT_CONF_THRESHOLD,
        "entropy_threshold": DEFAULT_ENTROPY_THRESHOLD,
        "coverage": float(1 - flagged.mean()),
        "flag_rate": float(flagged.mean()),
        "accuracy_on_accepted": float(correct[~flagged].mean()),
        "accuracy_on_flagged": float(correct[flagged].mean()) if flagged.any() else None,
        "errors_total": int((~correct).sum()),
        "errors_caught_by_gate": int((~correct & flagged).sum()),
        "auroc_msp_error_detection": float(roc_auc_score(~correct, -msp)) if (~correct).any() else None,
    }
    # risk-coverage curve by sweeping the confidence threshold
    sweep = []
    for tc in np.round(np.arange(0.30, 0.96, 0.05), 2):
        f = (msp < tc) | (ent > DEFAULT_ENTROPY_THRESHOLD)
        sweep.append({"conf_threshold": float(tc), "coverage": float(1 - f.mean()),
                      "accuracy_on_accepted": float(correct[~f].mean()) if (~f).any() else None})
    gate_in["threshold_sweep"] = sweep
    print(f"Gate on test: coverage {gate_in['coverage']:.3f}, accepted acc {gate_in['accuracy_on_accepted']:.4f}, "
          f"caught {gate_in['errors_caught_by_gate']}/{gate_in['errors_total']} errors")

    # ---------------- 3. Out-of-scope (OOD) detection ----------------
    ood = {}
    synth = synthetic_ood_images()
    p_syn = predict(model, device, images=synth)
    ood["synthetic_patterns"] = {"n": len(synth), "flag_rate": float(gate_mask(p_syn).mean()),
                                 "mean_msp": float(p_syn.max(1).mean())}
    ood_dir = ROOT / args.ood_dir
    if ood_dir.exists():
        paths = sorted(p for p in ood_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng = random.Random(SEED)
        paths = rng.sample(paths, min(args.ood_samples, len(paths)))
        p_ood = predict(model, device, paths=paths)
        f_ood = gate_mask(p_ood)
        labels = np.r_[np.zeros(len(probs)), np.ones(len(p_ood))]
        ood["imagenette"] = {
            "description": "Imagenette val (dogs, fish, church, parachute, cassette player, chainsaw, "
                           "French horn, golf ball, garbage truck, gas pump) - everyday non-waste objects",
            "n": len(paths),
            "flag_rate": float(f_ood.mean()),
            "mean_msp": float(p_ood.max(1).mean()),
            "auroc_msp": float(roc_auc_score(labels, -np.r_[msp, p_ood.max(1)])),
            "auroc_entropy": float(roc_auc_score(labels, np.r_[ent, normalized_entropy(p_ood)])),
            "most_common_wrong_class": pd.Series([class_names[i] for i in p_ood.argmax(1)]).value_counts().head(5).to_dict(),
        }
        print(f"OOD Imagenette: flagged {f_ood.mean():.3f}, AUROC(MSP) {ood['imagenette']['auroc_msp']:.3f}")
    else:
        print(f"[skip] OOD folder {ood_dir} not found")

    # ---------------- 4. Lighting / occlusion / glare / blur robustness ----------------
    sub = test.groupby("label").sample(n=args.robust_per_class, random_state=SEED)
    sub_paths, sub_y = [ROOT / p for p in sub.path], sub.label.to_numpy()
    sub_yc = to_category(sub_y, class_names)
    sweeps = {
        "gamma": [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 1.8, 2.2, 2.6],
        "occlusion_pct": [0, 10, 20, 30, 40, 50, 60, 70],
        "glare": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "blur": [0, 2, 4, 6, 8, 10],
    }
    robustness = {}
    for param, values in sweeps.items():
        rows = []
        for v in values:
            p = predict(model, device, paths=sub_paths, perturb={param: v})
            pred = p.argmax(1)
            f = gate_mask(p)
            ok = pred == sub_y
            rows.append({
                "value": v,
                "accuracy_12": float(ok.mean()),
                "accuracy_5": float((to_category(pred, class_names) == sub_yc).mean()),
                "flag_rate": float(f.mean()),
                "mean_msp": float(p.max(1).mean()),
                "mean_entropy": float(normalized_entropy(p).mean()),
                "accuracy_on_accepted": float(ok[~f].mean()) if (~f).any() else None,
            })
            print(f"  {param}={v}: acc {rows[-1]['accuracy_12']:.3f}, flagged {rows[-1]['flag_rate']:.3f}")
        robustness[param] = rows

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = {"gamma": "Lighting gamma (<1 dark, >1 bright)", "occlusion_pct": "Central occlusion (% area)",
              "glare": "Specular glare strength", "blur": "Blur radius (px @224)"}
    for ax, (param, rows) in zip(axes, robustness.items()):
        x = [r["value"] for r in rows]
        ax.plot(x, [r["accuracy_12"] for r in rows], "o-", label="12-class accuracy")
        ax.plot(x, [r["accuracy_5"] for r in rows], "s-", label="5-bin accuracy")
        ax.plot(x, [r["flag_rate"] for r in rows], "^--", label="flagged by gate")
        ax.plot(x, [r["accuracy_on_accepted"] or 0 for r in rows], "d:", label="accuracy on accepted")
        ax.set_title(titles[param])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "robustness_curves.png", dpi=140)
    plt.close(fig)

    # ---------------- 5. Failure & success galleries ----------------
    for d in ("failures", "successes"):
        shutil.rmtree(ARTIFACTS / d, ignore_errors=True)
        (ARTIFACTS / d).mkdir()
    wrong = np.where(~correct)[0]
    wrong = wrong[np.argsort(-msp[wrong])]  # most confident mistakes first
    gallery = {"failures": [], "successes": []}
    for rank, i in enumerate(wrong[:16]):
        name = f"{rank:02d}_{class_names[y_true[i]]}_as_{class_names[y_pred[i]]}.jpg"
        save_thumb(test_paths[i], ARTIFACTS / "failures" / name)
        gallery["failures"].append({
            "file": f"failures/{name}", "true": class_names[y_true[i]], "pred": class_names[y_pred[i]],
            "confidence": float(msp[i]), "entropy": float(ent[i]), "flagged": bool(flagged[i]),
            "same_bin": bool(yc_true[i] == yc_pred[i]),
        })
    rng = np.random.default_rng(SEED)
    for c in range(len(class_names)):
        idx = np.where(correct & (y_true == c))[0]
        if len(idx):
            i = int(rng.choice(idx))
            name = f"{class_names[c]}.jpg"
            save_thumb(test_paths[i], ARTIFACTS / "successes" / name)
            gallery["successes"].append({"file": f"successes/{name}", "true": class_names[c],
                                         "confidence": float(msp[i]), "entropy": float(ent[i])})

    report = {
        "model": meta,
        "device": device,
        "test_size": int(len(test)),
        "val_size": int(len(val)),
        "fine_grained": {"accuracy": rep12["accuracy"], "macro_avg": rep12["macro avg"],
                         "weighted_avg": rep12["weighted avg"],
                         "per_class": {c: rep12[c] for c in class_names},
                         "confusion_matrix": cm12.tolist(), "labels": class_names},
        "category": {"accuracy": rep5["accuracy"], "macro_avg": rep5["macro avg"],
                     "per_class": {c: rep5[c] for c in CATEGORY_NAMES},
                     "confusion_matrix": cm5.tolist(), "labels": CATEGORY_NAMES},
        "gate_in_distribution": gate_in,
        "ood": ood,
        "robustness": robustness,
        "gallery": gallery,
    }
    with open(ARTIFACTS / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved artifacts/evaluation_report.json")


if __name__ == "__main__":
    main()

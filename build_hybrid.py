"""
build_hybrid.py
Builds and evaluates the hybrid recogniser (see hybrid.py).

  1. SigLIP image embeddings + ConvNeXt probabilities for: Kaggle train/val/test,
     TrashNet (external), the second external garbage set, Imagenette (non-waste).
  2. Linear probe (logistic regression) on SigLIP embeddings of the TRAIN split.
  3. Fusion weights / temperature tuned on Kaggle VAL + TrashNet (tuning sets).
     Thresholds: out-of-scope and confidence thresholds chosen so >= 95% of in-scope
     tuning images are answered (no OOD data is used to set them).
  4. Held-out report: Kaggle TEST, external set #2, Imagenette, synthetic patterns,
     lighting / occlusion / glare / blur sweeps. Each expert alone vs. the fusion.

Outputs: artifacts/siglip_probe.npz, artifacts/hybrid_config.json, artifacts/hybrid_report.json
"""
import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from evaluate import predict as convnext_predict
from evaluate import synthetic_ood_images
from hybrid import (
    CONFIG_PATH,
    PROBE_PATH,
    SigLIPEncoder,
    category_matrix,
    decide,
    fuse,
    probe_probs,
    zero_shot_from_embeddings,
)
from waste_core import ARTIFACTS, CATEGORY_NAMES, ROOT, apply_perturbations, load_trained_model
from waste_vocab import NON_WASTE, WASTE_ITEMS

SEED = 42
FEAT_DIR = ARTIFACTS / "features"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# External dataset label -> our dataset class (category is what we score)
TRASHNET_MAP = {"cardboard": "cardboard", "glass": "white-glass", "metal": "metal", "paper": "paper",
                "plastic": "plastic", "trash": "trash"}


class SigDataset(Dataset):
    def __init__(self, paths, processor, perturb=None):
        self.paths, self.processor, self.perturb = list(paths), processor, perturb or {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        if self.perturb:
            im = apply_perturbations(im, **self.perturb)
        return self.processor(images=im, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def siglip_embed(enc: SigLIPEncoder, paths, perturb=None, batch_size=16):
    loader = DataLoader(SigDataset(paths, enc.processor, perturb), batch_size=batch_size, num_workers=2)
    out = []
    for px in loader:
        emb = enc.model.get_image_features(pixel_values=px.to(enc.device, enc.dtype)).float()
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out)


def features(name, paths, enc, cn, device, force=False):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    f = FEAT_DIR / f"{name}.npz"
    if f.exists() and not force:
        d = np.load(f)
        if len(d["emb"]) == len(paths):
            return d["emb"], d["cn"]
    print(f"  extracting {name} ({len(paths)} images)", flush=True)
    emb = siglip_embed(enc, paths)
    p_cn = convnext_predict(cn, device, paths=paths)
    np.savez_compressed(f, emb=emb, cn=p_cn)
    return emb, p_cn


def list_folder_dataset(root: Path, mapping: dict, class_names):
    paths, labels = [], []
    for src, dst in mapping.items():
        d = root / src
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                paths.append(p)
                labels.append(class_names.index(dst))
    return paths, np.array(labels)


def cat_of(labels, class_names):
    m = category_matrix(class_names)
    return m[labels].argmax(1)


def score(cat_probs, status, y_cat):
    pred = cat_probs.argmax(1)
    ok = status == "ok"
    return {
        "n": int(len(y_cat)),
        "accuracy": float((pred == y_cat).mean()),
        "macro_f1": float(f1_score(y_cat, pred, average="macro", labels=np.unique(y_cat))),
        "answered": float(ok.mean()),
        "accuracy_on_answered": float((pred[ok] == y_cat[ok]).mean()) if ok.any() else None,
        "out_of_scope_rate": float((status == "out_of_scope").mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust_per_class", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    cn, class_names, _, device = load_trained_model()
    enc = SigLIPEncoder(device)
    text_emb = enc.vocabulary_embeddings()
    M = category_matrix(class_names)

    split = pd.read_csv(ARTIFACTS / "split.csv")
    sets = {}
    for s in ("train", "val", "test"):
        d = split[split.split == s]
        sets[f"kaggle_{s}"] = ([ROOT / p for p in d.path], d.label.to_numpy())
    tp, tl = list_folder_dataset(ROOT / "data" / "trashnet" / "dataset-resized", TRASHNET_MAP, class_names)
    if tp:
        sets["trashnet"] = (tp, tl)
    ext2 = json.loads((ROOT / "data" / "external2" / "mapping.json").read_text()) \
        if (ROOT / "data" / "external2" / "mapping.json").exists() else None
    if ext2:
        p2, l2 = list_folder_dataset(ROOT / "data" / "external2", ext2, class_names)
        sets["external2"] = (p2, l2)
    ood_dir = ROOT / "data" / "imagenette2-160" / "val"
    ood_paths = sorted(p for p in ood_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    ood_paths = random.Random(SEED).sample(ood_paths, min(1000, len(ood_paths)))

    feats = {k: features(k, v[0], enc, cn, device, args.force) for k, v in sets.items()}
    feats["imagenette"] = features("imagenette", ood_paths, enc, cn, device, args.force)

    # ---------------- linear probe on TRAIN embeddings
    Xtr, ytr = feats["kaggle_train"][0], sets["kaggle_train"][1]
    Xva, yva = feats["kaggle_val"][0], sets["kaggle_val"][1]
    best = None
    for C in [0.5, 1, 2, 5, 10]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        acc = (clf.predict(Xva) == yva).mean()
        print(f"  probe C={C}: val acc {acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    probe = {"coef": best[2].coef_.astype(np.float32), "intercept": best[2].intercept_.astype(np.float32)}
    np.savez(PROBE_PATH, **probe)
    print(f"Linear probe: C={best[1]}, val acc {best[0]:.4f}")

    def experts(name, T):
        emb, p_cn = feats[name]
        _, p_zs, nw = zero_shot_from_embeddings(emb, text_emb, enc.logit_scale, class_names, T)
        return p_zs, probe_probs(emb, probe), p_cn, nw

    # ---------------- tune fusion on kaggle_val + trashnet
    tune_sets = [s for s in ("kaggle_val", "trashnet") if s in feats]
    grid = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3) if abs(sum(w) - 1) < 1e-6]
    results = []
    for T in [0.5, 1.0, 2.0]:
        cache = {s: experts(s, T) for s in tune_sets}
        for w in grid:
            cfg = {"w_zs": w[0], "w_lp": w[1], "w_cn": w[2]}
            accs = []
            for s in tune_sets:
                p_zs, p_lp, p_cn, _ = cache[s]
                pred = (fuse(p_zs, p_lp, p_cn, cfg) @ M).argmax(1)
                accs.append((pred == cat_of(sets[s][1], class_names)).mean())
            results.append((float(np.mean(accs)), T, w, [float(a) for a in accs]))
    results.sort(key=lambda r: -r[0])
    top_acc, T, w, accs = results[0]
    print(f"Best fusion: T={T} w(zs,lp,cn)={w} tuning acc {dict(zip(tune_sets, accs))}")
    cfg = {"w_zs": float(w[0]), "w_lp": float(w[1]), "w_cn": float(w[2]), "temperature": float(T)}

    # thresholds from in-scope tuning data only
    nw_all, conf_all, ent_all = [], [], []
    for s in tune_sets:
        p_zs, p_lp, p_cn, nw = experts(s, T)
        cat = fuse(p_zs, p_lp, p_cn, cfg) @ M
        nw_all.append(nw)
        conf_all.append(cat.max(1))
        ent_all.append(-(cat * np.log(cat + 1e-12)).sum(1) / np.log(cat.shape[1]))
    nw_all, conf_all, ent_all = map(np.concatenate, (nw_all, conf_all, ent_all))
    cfg["ood_threshold"] = float(np.clip(np.quantile(nw_all, 0.97), 0.3, 0.8))
    cfg["conf_threshold"] = float(np.clip(np.quantile(conf_all, 0.03), 0.4, 0.7))
    cfg["entropy_threshold"] = float(np.clip(np.quantile(ent_all, 0.97), 0.4, 0.8))
    cfg["tuning_sets"] = tune_sets
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Config:", cfg)

    # ---------------- held-out evaluation: hybrid vs each expert
    report = {"config": cfg, "probe_C": best[1], "probe_val_acc": float(best[0]),
              "vocabulary": {"waste_items": len(WASTE_ITEMS), "non_waste_concepts": len(NON_WASTE), "temperature": T}, "sets": {}}
    eval_sets = [s for s in ("kaggle_test", "trashnet", "external2") if s in feats]
    for s in eval_sets:
        p_zs, p_lp, p_cn, nw = experts(s, T)
        y_cat = cat_of(sets[s][1], class_names)
        entry = {"held_out": s != "trashnet"}
        for name, p in [("hybrid", fuse(p_zs, p_lp, p_cn, cfg)), ("zero_shot_only", p_zs),
                        ("probe_only", p_lp), ("convnext_only", p_cn)]:
            cat, status = decide(p, nw if name in ("hybrid", "zero_shot_only") else np.zeros(len(nw)),
                                 class_names, cfg)
            entry[name] = score(cat, status, y_cat)
            if name == "hybrid":
                entry["confusion_matrix"] = confusion_matrix(y_cat, cat.argmax(1), labels=range(5)).tolist()
        report["sets"][s] = entry
        print(f"{s}: hybrid {entry['hybrid']['accuracy']:.4f} | convnext {entry['convnext_only']['accuracy']:.4f} | "
              f"zero-shot {entry['zero_shot_only']['accuracy']:.4f} | probe {entry['probe_only']['accuracy']:.4f} | "
              f"answered {entry['hybrid']['answered']:.3f}")

    # out-of-scope
    p_zs, p_lp, p_cn, nw = experts("imagenette", T)
    _, status = decide(fuse(p_zs, p_lp, p_cn, cfg), nw, class_names, cfg)
    syn = synthetic_ood_images()
    emb_syn = enc.encode_images(syn)
    _, pz, nws = zero_shot_from_embeddings(emb_syn, text_emb, enc.logit_scale, class_names, T)
    pc = convnext_predict(cn, device, images=syn)
    _, st_syn = decide(fuse(pz, probe_probs(emb_syn, probe), pc, cfg), nws, class_names, cfg)
    report["ood"] = {
        "imagenette": {"n": len(status), "out_of_scope": float((status == "out_of_scope").mean()),
                       "not_answered": float((status != "ok").mean())},
        "synthetic": {"n": len(st_syn), "out_of_scope": float((st_syn == "out_of_scope").mean()),
                      "not_answered": float((st_syn != "ok").mean())},
    }
    print("OOD:", report["ood"])

    # robustness of the hybrid
    test = split[split.split == "test"]
    sub = test.groupby("label").sample(n=args.robust_per_class, random_state=SEED)
    sub_paths, sub_y = [ROOT / p for p in sub.path], cat_of(sub.label.to_numpy(), class_names)
    sweeps = {"gamma": [0.3, 0.5, 1.0, 1.8, 2.5], "occlusion_pct": [0, 20, 40, 60],
              "glare": [0.0, 0.5, 1.0], "blur": [0, 4, 8]}
    rob = {}
    for param, values in sweeps.items():
        rows = []
        for v in values:
            pert = {param: v}
            emb = siglip_embed(enc, sub_paths, pert)
            _, pz, nwr = zero_shot_from_embeddings(emb, text_emb, enc.logit_scale, class_names, T)
            pc = convnext_predict(cn, device, paths=sub_paths, perturb=pert)
            cat_h, st_h = decide(fuse(pz, probe_probs(emb, probe), pc, cfg), nwr, class_names, cfg)
            cat_c = pc @ M
            ok = st_h == "ok"
            rows.append({"value": v,
                         "hybrid_accuracy": float((cat_h.argmax(1) == sub_y).mean()),
                         "convnext_accuracy": float((cat_c.argmax(1) == sub_y).mean()),
                         "hybrid_answered": float(ok.mean()),
                         "hybrid_accuracy_on_answered": float((cat_h.argmax(1)[ok] == sub_y[ok]).mean()) if ok.any() else None})
            print(f"  {param}={v}: hybrid {rows[-1]['hybrid_accuracy']:.3f} convnext {rows[-1]['convnext_accuracy']:.3f}")
        rob[param] = rows
    report["robustness"] = rob
    (ARTIFACTS / "hybrid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved artifacts/hybrid_report.json")


if __name__ == "__main__":
    main()

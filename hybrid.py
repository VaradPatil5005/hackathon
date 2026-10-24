"""
hybrid.py
Hybrid waste recogniser used by the app and the evaluation scripts.

Three experts vote over the 12 dataset classes:
  1. SigLIP zero-shot  - open vocabulary: ~200 described waste items + ~50 "not waste"
                         concepts. Understands items never seen in the training data
                         (chocolate wrappers, cake, tetra packs, kulhads, ...).
  2. SigLIP linear probe - logistic regression on SigLIP image embeddings, trained on
                         the 70% training split.
  3. ConvNeXt-Tiny     - fully fine-tuned on the training split (train.py).

Fused 12-class probabilities are summed into the 5 cafeteria categories. The image is
out of scope if the zero-shot "not waste" mass is high, and uncertain if the fused
category confidence is low or its entropy is high. Weights / thresholds are tuned by
build_hybrid.py and stored in artifacts/hybrid_config.json.
"""
import hashlib
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from waste_core import (
    ARTIFACTS,
    CATEGORY_NAMES,
    CLASS_INFO,
    load_trained_model,
    normalized_entropy,
    predict_probs,
)
from waste_vocab import NON_WASTE, TEMPLATES, TIPS, WASTE_ITEMS

VLM_ID = "google/siglip-so400m-patch14-384"
CONFIG_PATH = ARTIFACTS / "hybrid_config.json"
PROBE_PATH = ARTIFACTS / "siglip_probe.npz"
TEXT_CACHE = ARTIFACTS / "siglip_text_emb.npz"

DEFAULT_CONFIG = {
    "w_zs": 0.4, "w_lp": 0.3, "w_cn": 0.3, "temperature": 1.0,
    "ood_threshold": 0.5, "conf_threshold": 0.6, "entropy_threshold": 0.6,
}


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def category_matrix(class_names: List[str]) -> np.ndarray:
    """(12, 5) 0/1 matrix mapping dataset classes to cafeteria categories."""
    m = np.zeros((len(class_names), len(CATEGORY_NAMES)))
    for i, c in enumerate(class_names):
        m[i, CATEGORY_NAMES.index(CLASS_INFO[c]["category"])] = 1
    return m


class SigLIPEncoder:
    """Image / text encoder with cached text embeddings for the waste vocabulary."""

    def __init__(self, device: Optional[str] = None):
        from transformers import AutoModel, AutoProcessor
        from waste_core import get_device

        self.device = get_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(VLM_ID)
        try:
            self.model = AutoModel.from_pretrained(VLM_ID, dtype=dtype).eval().to(self.device)
        except TypeError:
            self.model = AutoModel.from_pretrained(VLM_ID, torch_dtype=dtype).eval().to(self.device)
        self.dtype = dtype
        self.logit_scale = float(self.model.logit_scale.exp().item())

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            px = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            emb = self.model.get_image_features(pixel_values=px).float()
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.processor(text=texts[i:i + batch_size], padding="max_length", truncation=True,
                                 return_tensors="pt").to(self.device)
            emb = self.model.get_text_features(**tok).float()
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def vocabulary_embeddings(self) -> np.ndarray:
        """One embedding per item (template-averaged): WASTE_ITEMS followed by NON_WASTE."""
        names = [w[0] for w in WASTE_ITEMS] + NON_WASTE
        key = hashlib.sha1(json.dumps([names, TEMPLATES, VLM_ID]).encode()).hexdigest()
        if TEXT_CACHE.exists():
            cached = np.load(TEXT_CACHE)
            if str(cached["key"]) == key:
                return cached["emb"]
        texts = [t.format(n) for n in names for t in TEMPLATES]
        emb = self.encode_texts(texts).reshape(len(names), len(TEMPLATES), -1).mean(1)
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
        ARTIFACTS.mkdir(exist_ok=True)
        np.savez(TEXT_CACHE, emb=emb, key=key)
        return emb


def zero_shot_from_embeddings(img_emb: np.ndarray, text_emb: np.ndarray, logit_scale: float,
                              class_names: List[str], temperature: float = 1.0):
    """
    Returns (item_probs over all vocabulary items, class_probs (N,12) conditional on being waste,
    non_waste_mass (N,)).
    """
    logits = logit_scale * img_emb @ text_emb.T / temperature
    item_probs = softmax(logits)
    n_w = len(WASTE_ITEMS)
    waste = item_probs[:, :n_w]
    non_waste = item_probs[:, n_w:].sum(1)
    cls_idx = np.array([class_names.index(c) for _, c, _ in WASTE_ITEMS])
    class_probs = np.zeros((len(img_emb), len(class_names)))
    for j in range(len(class_names)):
        class_probs[:, j] = waste[:, cls_idx == j].sum(1)
    class_probs /= class_probs.sum(1, keepdims=True).clip(min=1e-12)
    return item_probs, class_probs, non_waste


def probe_probs(img_emb: np.ndarray, probe: Dict[str, np.ndarray]) -> np.ndarray:
    return softmax(img_emb @ probe["coef"].T + probe["intercept"])


def fuse(p_zs: np.ndarray, p_lp: np.ndarray, p_cn: np.ndarray, cfg: Dict) -> np.ndarray:
    p = cfg["w_zs"] * p_zs + cfg["w_lp"] * p_lp + cfg["w_cn"] * p_cn
    return p / p.sum(-1, keepdims=True)


def decide(fused12: np.ndarray, non_waste: np.ndarray, class_names: List[str], cfg: Dict):
    """Vectorised decision: returns (cat_probs (N,5), status array of 'ok'/'uncertain'/'out_of_scope')."""
    cat = fused12 @ category_matrix(class_names)
    conf = cat.max(1)
    ent = normalized_entropy(cat)
    status = np.where(non_waste > cfg["ood_threshold"], "out_of_scope",
                      np.where((conf < cfg["conf_threshold"]) | (ent > cfg["entropy_threshold"]),
                               "uncertain", "ok"))
    return cat, status


class HybridClassifier:
    def __init__(self, device: Optional[str] = None):
        self.cn_model, self.class_names, self.cn_meta, self.device = load_trained_model(device=device)
        self.cfg = dict(DEFAULT_CONFIG)
        if CONFIG_PATH.exists():
            self.cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        self.vlm = SigLIPEncoder(self.device)
        self.text_emb = self.vlm.vocabulary_embeddings()
        self.probe = dict(np.load(PROBE_PATH)) if PROBE_PATH.exists() else None
        self.cat_m = category_matrix(self.class_names)

    def predict(self, img: Image.Image, cfg: Optional[Dict] = None) -> Dict:
        cfg = {**self.cfg, **(cfg or {})}
        t0 = time.perf_counter()
        emb = self.vlm.encode_images([img])
        item_probs, p_zs, non_waste = zero_shot_from_embeddings(
            emb, self.text_emb, self.vlm.logit_scale, self.class_names, cfg["temperature"])
        p_lp = probe_probs(emb, self.probe) if self.probe is not None else p_zs
        p_cn = predict_probs(self.cn_model, [img], self.device)
        fused = fuse(p_zs, p_lp, p_cn, cfg)
        cat, status = decide(fused, non_waste, self.class_names, cfg)
        latency = (time.perf_counter() - t0) * 1000

        cat, fused, item_probs = cat[0], fused[0], item_probs[0]
        ci = int(cat.argmax())
        category = CATEGORY_NAMES[ci]
        # Best fine class inside the chosen category, then the best described item of that class.
        in_cat = [i for i, c in enumerate(self.class_names) if CLASS_INFO[c]["category"] == category]
        fine = self.class_names[max(in_cat, key=lambda i: fused[i])]
        n_w = len(WASTE_ITEMS)
        cand = [k for k in range(n_w) if WASTE_ITEMS[k][1] == fine] or list(range(n_w))
        k_best = max(cand, key=lambda k: item_probs[k])
        item_name, _, tip_key = WASTE_ITEMS[k_best]

        top_items = sorted(range(n_w), key=lambda k: -item_probs[k])[:5]
        nw_top = int(np.argmax(item_probs[n_w:]))
        h5 = float(normalized_entropy(cat))
        return {
            "status": str(status[0]),
            "category": category,
            "category_index": ci,
            "confidence": float(cat[ci]),
            "entropy": h5,
            "non_waste": float(non_waste[0]),
            "non_waste_guess": NON_WASTE[nw_top],
            "fine_class": fine,
            "item": item_name,
            "item_tip": TIPS.get(tip_key) if tip_key else None,
            "category_probs": {c: float(p) for c, p in zip(CATEGORY_NAMES, cat)},
            "class_probs": {c: float(p) for c, p in zip(self.class_names, fused)},
            "experts": {
                "zero_shot": {c: float(p) for c, p in zip(CATEGORY_NAMES, (p_zs @ self.cat_m)[0])},
                "linear_probe": {c: float(p) for c, p in zip(CATEGORY_NAMES, (p_lp @ self.cat_m)[0])},
                "convnext": {c: float(p) for c, p in zip(CATEGORY_NAMES, (p_cn @ self.cat_m)[0])},
            },
            "top_items": [{"item": WASTE_ITEMS[k][0], "p": float(item_probs[k]),
                           "category": CLASS_INFO[WASTE_ITEMS[k][1]]["category"]} for k in top_items],
            "latency_ms": latency,
            "thresholds": {k: cfg[k] for k in ("ood_threshold", "conf_threshold", "entropy_threshold")},
        }

"""
waste_core.py
Shared logic for the Smart Waste Segregation Assistant: category mapping,
disposal instructions, image perturbations, model loading and the
MSP + normalized-entropy uncertainty gate. Used by app.py, train.py and
evaluate.py so that the demo and the reported numbers use identical code.
"""
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent

# Keep every download / cache inside the project folder.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import models, transforms  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# 5 cafeteria disposal categories
# ---------------------------------------------------------------------------
CATEGORIES: Dict[str, Dict[str, str]] = {
    "Paper & Cardboard": {
        "bin": "Blue Bin - Dry Paper Recyclables",
        "color": "#2f7f9e",
        "icon": "📄",
    },
    "Plastic": {
        "bin": "Yellow Bin - Recyclable Plastics",
        "color": "#d99a06",
        "icon": "🧴",
    },
    "Metal": {
        "bin": "Grey Bin - Metals & Cans",
        "color": "#6e6e66",
        "icon": "🥫",
    },
    "Food Scraps / Organic": {
        "bin": "Green Bin - Wet / Compostable Food Waste",
        "color": "#4e7a1a",
        "icon": "🍎",
    },
    "Non-Recyclable / Special Waste": {
        "bin": "Red/Black Bin or Special-Handling Point",
        "color": "#b4472b",
        "icon": "🗑️",
    },
}
CATEGORY_NAMES: List[str] = list(CATEGORIES.keys())

# ---------------------------------------------------------------------------
# 12 fine-grained dataset classes -> 5 cafeteria categories + instructions
# (keys are the dataset folder names, lower-case)
# ---------------------------------------------------------------------------
CLASS_INFO: Dict[str, Dict[str, str]] = {
    "paper": {
        "display": "Paper",
        "category": "Paper & Cardboard",
        "target": "Blue Bin",
        "instruction": (
            "1. Make sure the paper (cups, sleeves, sheets, bags) is dry and free of heavy food residue.\n"
            "2. Flatten it and drop it into the BLUE BIN.\n"
            "3. If it is soaked in oil or curry, it is no longer recyclable - put it in the GREEN (food) bin."
        ),
    },
    "cardboard": {
        "display": "Cardboard",
        "category": "Paper & Cardboard",
        "target": "Blue Bin",
        "instruction": (
            "1. Shake out leftover food crumbs from takeaway boxes.\n"
            "2. Tear off heavily greased parts, flatten the box and place it in the BLUE BIN."
        ),
    },
    "plastic": {
        "display": "Plastic",
        "category": "Plastic",
        "target": "Yellow Bin",
        "instruction": (
            "1. Empty all liquid from bottles, cups and containers.\n"
            "2. Rinse lightly if a sink is nearby, crush to save space, put the cap back on.\n"
            "3. Drop it into the YELLOW BIN."
        ),
    },
    "metal": {
        "display": "Metal",
        "category": "Metal",
        "target": "Grey Bin",
        "instruction": (
            "1. Drain cans and tins completely.\n"
            "2. Wipe food off aluminium foil and ball it up.\n"
            "3. Place it in the GREY METAL BIN."
        ),
    },
    "biological": {
        "display": "Food / Biological",
        "category": "Food Scraps / Organic",
        "target": "Green Bin",
        "instruction": (
            "1. Put leftover rice, chapati, peels, vegetables and tea bags in the GREEN BIN.\n"
            "2. IMPORTANT: remove plastic wrappers, foil, toothpicks and cutlery first."
        ),
    },
    "trash": {
        "display": "General Trash",
        "category": "Non-Recyclable / Special Waste",
        "target": "Red/Black Bin",
        "instruction": (
            "1. Multi-layer chip packets, styrofoam, used tissues and straws go in the RED/BLACK BIN.\n"
            "2. Keep these out of the Blue, Yellow and Green bins - they contaminate recycling."
        ),
    },
    "battery": {
        "display": "Battery",
        "category": "Non-Recyclable / Special Waste",
        "target": "E-Waste / Battery Drop Box",
        "instruction": (
            "1. FIRE HAZARD - never put batteries in any cafeteria bin.\n"
            "2. Hand it over at the campus E-Waste / Battery drop box or to cafeteria staff."
        ),
    },
    "brown-glass": {
        "display": "Brown Glass",
        "category": "Non-Recyclable / Special Waste",
        "target": "Glass Collection Crate",
        "instruction": (
            "1. Handle carefully. Place intact bottles in the glass collection crate.\n"
            "2. If it is broken, do not touch the shards - inform cafeteria staff."
        ),
    },
    "green-glass": {
        "display": "Green Glass",
        "category": "Non-Recyclable / Special Waste",
        "target": "Glass Collection Crate",
        "instruction": (
            "1. Empty the bottle and place it in the glass collection crate.\n"
            "2. Never mix glass (especially broken glass) with paper or plastic."
        ),
    },
    "white-glass": {
        "display": "Clear Glass",
        "category": "Non-Recyclable / Special Waste",
        "target": "Glass Collection Crate",
        "instruction": (
            "1. Rinse clear jars/bottles and place them in the glass collection crate.\n"
            "2. If shattered, ask staff for the puncture-safe container."
        ),
    },
    "clothes": {
        "display": "Clothes / Textile",
        "category": "Non-Recyclable / Special Waste",
        "target": "Red/Black Bin or Textile Drive",
        "instruction": (
            "1. Fabric does not belong in food or recycling bins.\n"
            "2. Soiled rags -> RED/BLACK BIN. Clean wearable items -> hostel donation box."
        ),
    },
    "shoes": {
        "display": "Shoes",
        "category": "Non-Recyclable / Special Waste",
        "target": "Red/Black Bin or Donation Point",
        "instruction": (
            "1. Footwear is composite waste (rubber/leather/fabric) and cannot be processed in cafeteria streams.\n"
            "2. Use the bulky non-recyclable bin, or donate if wearable."
        ),
    },
}

UNCERTAIN_ADVICE = (
    "1. **Check the packaging label** for a recycling symbol (PET 1, HDPE 2, PAP 20-22, ALU 41).\n"
    "2. **Retake the photo**: one item, centred, hand out of the way, no harsh glare.\n"
    "3. **Ask cafeteria staff** for help. If no one is available, use the Red/Black bin so that "
    "recyclable streams are not contaminated."
)

DEFAULT_CONF_THRESHOLD = 0.65
DEFAULT_ENTROPY_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_trained_model(path: Path = MODEL_PATH, device: Optional[str] = None):
    """Returns (model, class_names, meta, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    class_names = ckpt["class_names"]
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, class_names, ckpt.get("meta", {}), device


# ---------------------------------------------------------------------------
# Uncertainty metrics
# ---------------------------------------------------------------------------
def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy divided by ln(K), in [0, 1]. Works on (K,) or (N, K)."""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.shape[-1]
    h = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    return h / math.log(k)


def aggregate_to_categories(probs: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Sums fine-grained probabilities into the 5 cafeteria categories. (N,12)->(N,5)."""
    probs = np.atleast_2d(probs)
    out = np.zeros((probs.shape[0], len(CATEGORY_NAMES)))
    for j, name in enumerate(class_names):
        out[:, CATEGORY_NAMES.index(CLASS_INFO[name]["category"])] += probs[:, j]
    return out


def gate(max_prob: float, h_norm: float, conf_threshold: float, entropy_threshold: float):
    """Returns (is_uncertain, reasons)."""
    reasons = []
    if max_prob < conf_threshold:
        reasons.append(f"top confidence {max_prob:.1%} is below the {conf_threshold:.0%} threshold")
    if h_norm > entropy_threshold:
        reasons.append(f"normalized entropy {h_norm:.2f} is above the {entropy_threshold:.2f} limit")
    return bool(reasons), reasons


@torch.no_grad()
def predict_probs(model, images: List[Image.Image], device: str, image_size: int = 224) -> np.ndarray:
    tf = eval_transform(image_size)
    batch = torch.stack([tf(img.convert("RGB")) for img in images]).to(device)
    with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        logits = model(batch)
    return F.softmax(logits.float(), dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Lighting / occlusion / blur perturbations (shared by app and evaluation)
# ---------------------------------------------------------------------------
def apply_perturbations(
    pil_img: Image.Image,
    gamma: float = 1.0,
    glare: float = 0.0,
    occlusion_pct: float = 0.0,
    blur: int = 0,
) -> Image.Image:
    """
    gamma < 1 darkens (dim corner), gamma > 1 brightens (over-exposure).
    glare: strength of a gaussian specular highlight (0-1).
    occlusion_pct: % of image area covered by a central dark patch (hand / napkin).
    blur: gaussian blur radius in pixels (motion / focus blur).
    """
    img = np.array(pil_img.convert("RGB"))
    if abs(gamma - 1.0) > 1e-3:
        table = (((np.arange(256) / 255.0) ** (1.0 / max(gamma, 0.05))) * 255).clip(0, 255).astype("uint8")
        img = cv2.LUT(img, table)
    if glare > 0:
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy, sigma = w * 0.55, h * 0.40, min(h, w) * 0.28
        mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))[..., None]
        img = np.clip(img.astype(np.float32) + glare * 255.0 * mask, 0, 255).astype(np.uint8)
    if occlusion_pct > 0:
        h, w = img.shape[:2]
        side_h = int(h * math.sqrt(occlusion_pct / 100.0))
        side_w = int(w * math.sqrt(occlusion_pct / 100.0))
        y1, x1 = (h - side_h) // 2, (w - side_w) // 2
        img = img.copy()
        img[y1 : y1 + side_h, x1 : x1 + side_w] = (45, 42, 38)
    if blur > 0:
        # blur is expressed at 224 px (model input); scale to the actual image size
        radius = max(1, int(round(blur * min(img.shape[:2]) / 224)))
        k = radius * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    return Image.fromarray(img)

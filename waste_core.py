"""
waste_core.py
Shared logic for the Smart Waste Segregation Assistant: category mapping,
disposal instructions, image perturbations, model loading and the
MSP + normalized-entropy uncertainty gate. Used by app.py, train.py and
evaluate.py so that the demo and the reported numbers use identical code.
"""
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
POLICY_PATH = ROOT / "policy.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Policy Loading (Editable without retraining)
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Paper & Cardboard": {
        "bin": "Blue Bin - Dry Paper Recyclables",
        "color": "#2f7f9e",
        "icon": "📄",
        "description": "Clean, dry paper and cardboard free of heavy grease or food residue.",
    },
    "Plastic": {
        "bin": "Yellow Bin - Recyclable Plastics",
        "color": "#d99a06",
        "icon": "🧴",
        "description": "Clean, rigid recyclable plastic containers (primarily PET #1 and HDPE #2).",
    },
    "Metal": {
        "bin": "Grey Bin - Metals & Cans",
        "color": "#6e6e66",
        "icon": "🥫",
        "description": "Beverage cans, clean tins, and metal lids. Emptied and rinsed.",
    },
    "Food Scraps / Organic": {
        "bin": "Green Bin - Wet / Compostable Food Waste",
        "color": "#4e7a1a",
        "icon": "🍎",
        "description": "Post-consumer food scraps and compostable solids. Strictly no non-biodegradable items.",
    },
    "Non-Recyclable / Special Waste": {
        "bin": "Red/Black Bin or Special-Handling Point",
        "color": "#b4472b",
        "icon": "🗑️",
        "description": "Composite category combining general residual waste (Red/Black bin) with special-handling points (batteries, glass crates, donations). This is NOT a single physical bin.",
    },
}

DEFAULT_CLASS_INFO: Dict[str, Dict[str, str]] = {
    "paper": {
        "display": "Paper",
        "category": "Paper & Cardboard",
        "target": "Blue Bin",
        "instruction": (
            "1. Make sure paper (cups, sleeves, sheets, bags) is clean and dry.\n"
            "2. Flatten and place in the BLUE BIN.\n"
            "3. Greasy or food-soiled paper cannot be recycled in the blue stream; place heavily soiled paper in residual waste (Red/Black Bin) unless your facility specifically composts soiled paper."
        ),
    },
    "cardboard": {
        "display": "Cardboard",
        "category": "Paper & Cardboard",
        "target": "Blue Bin",
        "instruction": (
            "1. Shake out all leftover food crumbs from takeaway boxes.\n"
            "2. Flatten clean cardboard and place it in the BLUE BIN.\n"
            "3. Tear off heavily greased sections and discard them into residual waste (Red/Black Bin)."
        ),
    },
    "plastic": {
        "display": "Plastic",
        "category": "Plastic",
        "target": "Yellow Bin",
        "instruction": (
            "1. Empty all liquid or food residues completely.\n"
            "2. Check for recycling codes (#1 PET bottles, #2 HDPE containers). Lightly rinse and drop in the YELLOW BIN.\n"
            "3. Note: thin plastic films, unmarked containers, and plastic straws belong in the Red/Black residual bin."
        ),
    },
    "metal": {
        "display": "Metal",
        "category": "Metal",
        "target": "Grey Bin",
        "instruction": (
            "1. Drain cans and tins completely.\n"
            "2. Wipe food residue off aluminium foil and crumple into a ball.\n"
            "3. Place in the GREY METAL BIN."
        ),
    },
    "biological": {
        "display": "Food / Biological",
        "category": "Food Scraps / Organic",
        "target": "Green Bin",
        "instruction": (
            "1. Put leftover rice, chapati, fruit peels, vegetables, and tea bags in the GREEN BIN.\n"
            "2. CRITICAL: Remove all plastic wrappers, foil, toothpicks, sachets, and cutlery first to avoid batch contamination."
        ),
    },
    "trash": {
        "display": "General Trash",
        "category": "Non-Recyclable / Special Waste",
        "target": "Red/Black Bin",
        "instruction": (
            "1. Multi-layer snack wrappers, styrofoam containers, soiled tissues, and composite waste go in the RED/BLACK BIN.\n"
            "2. Keep these out of the Blue, Yellow, and Green bins to protect recyclables from contamination."
        ),
    },
    "battery": {
        "display": "Battery",
        "category": "Non-Recyclable / Special Waste",
        "target": "E-Waste / Battery Drop Box",
        "instruction": (
            "1. FIRE & CHEMICAL HAZARD: NEVER discard batteries into any general cafeteria bin.\n"
            "2. Hand directly to cafeteria staff or deposit in the designated campus E-Waste / Battery Drop Box."
        ),
    },
    "brown-glass": {
        "display": "Brown Glass",
        "category": "Non-Recyclable / Special Waste",
        "target": "Glass Collection Crate",
        "instruction": (
            "1. Rinse bottles and place intact glass in the dedicated Glass Collection Crate (kept separate from commingled recyclables to prevent shatter hazard).\n"
            "2. If glass is broken, do not touch shards barehanded; immediately notify cafeteria staff."
        ),
    },
    "green-glass": {
        "display": "Green Glass",
        "category": "Non-Recyclable / Special Waste",
        "target": "Glass Collection Crate",
        "instruction": (
            "1. Empty liquids and place intact bottles in the dedicated Glass Collection Crate.\n"
            "2. Never mix glass bottles with paper or plastic bins."
        ),
    },
    "white-glass": {
        "display": "Clear Glass",
        "category": "Non-Recyclable / Special Waste",
        "target": "Glass Collection Crate",
        "instruction": (
            "1. Rinse clear glass jars/bottles and place them in the dedicated Glass Collection Crate.\n"
            "2. For shattered glassware, ask staff for puncture-resistant disposal."
        ),
    },
    "clothes": {
        "display": "Clothes / Textile",
        "category": "Non-Recyclable / Special Waste",
        "target": "Textile Donation Box or Red/Black Bin",
        "instruction": (
            "1. Textiles do not belong in standard cafeteria recycling or food bins.\n"
            "2. Clean, wearable items should go to a campus donation point; heavily soiled rags go to the RED/BLACK BIN."
        ),
    },
    "shoes": {
        "display": "Shoes",
        "category": "Non-Recyclable / Special Waste",
        "target": "Bulky Residual Bin or Donation Point",
        "instruction": (
            "1. Footwear contains composite materials (rubber, adhesives, leather, synthetic fabric) unsuitable for cafeteria recycling streams.\n"
            "2. Donate wearable shoes or place damaged footwear into bulky residual waste."
        ),
    },
}

def load_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads editable policy configuration from JSON with fallback."""
    p = path or POLICY_PATH
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data
        except Exception:
            pass
    return {
        "demonstration_notice": "Bin colours, collection streams, and disposal guidance are demonstration assumptions based on standard cafeteria guidelines. Campus and municipal policies vary; consult your local institution's waste management rules.",
        "categories": DEFAULT_CATEGORIES,
        "classes": DEFAULT_CLASS_INFO,
    }

POLICY = load_policy()
CATEGORIES: Dict[str, Dict[str, Any]] = POLICY.get("categories", DEFAULT_CATEGORIES)
CATEGORY_NAMES: List[str] = list(CATEGORIES.keys())
CLASS_INFO: Dict[str, Dict[str, str]] = POLICY.get("classes", DEFAULT_CLASS_INFO)
DEMONSTRATION_NOTICE: str = POLICY.get(
    "demonstration_notice",
    "Bin colours and collection rules are demonstration assumptions; check local campus guidelines.",
)

UNCERTAIN_ADVICE = (
    "1. **Check the packaging label** for a recycling symbol (e.g. PET 1, HDPE 2, PAP 20-22, ALU 41).\n"
    "2. **Retake the photo**: one item, centered, unobstructed, with even lighting.\n"
    "3. **Consult cafeteria staff or signage**. Do not guess or automatically discard unverified items into recycling streams."
)

DEFAULT_CONF_THRESHOLD = 0.65
DEFAULT_ENTROPY_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Device Resolution (Apple MPS, CUDA, CPU fallback)
# ---------------------------------------------------------------------------
def get_device(requested: Optional[str] = None) -> str:
    """
    Returns verified execution device: 'mps', 'cuda', or 'cpu'.
    Can be configured via BINZY_DEVICE environment variable or argument.
    """
    req = (requested or os.environ.get("BINZY_DEVICE", "")).strip().lower()
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    if req == "mps" and torch.backends.mps.is_available():
        return "mps"
    if req == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    """Returns (model, class_names, meta, device) with MPS/CPU/CUDA support."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{path}'. "
            "Please copy the trained 'model.pt' to the 'artifacts/' directory. "
            "Run 'python preflight.py' to inspect prerequisites."
        )
    target_device = get_device(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    class_names = ckpt["class_names"]
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(target_device)
    return model, class_names, ckpt.get("meta", {}), target_device


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
    if device == "cuda":
        with torch.autocast(device_type="cuda"):
            logits = model(batch)
    else:
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

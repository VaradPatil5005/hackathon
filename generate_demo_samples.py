"""
generate_demo_samples.py
Builds an offline demo_samples/ folder from the held-out TEST split (never seen
in training): one clean image per class, degraded variants (dark, glare,
occluded) and out-of-scope images (Imagenette photos + a synthetic pattern).

Usage:  python generate_demo_samples.py
"""
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from waste_core import ARTIFACTS, CLASS_INFO, ROOT, apply_perturbations

OUT = ROOT / "demo_samples"
SEED = 3


def save(img: Image.Image, name: str):
    img = img.convert("RGB")
    img.thumbnail((640, 640))
    img.save(OUT / f"{name}.jpg", quality=92)


def main():
    split_path = ARTIFACTS / "split.csv"
    if not split_path.exists():
        raise SystemExit("artifacts/split.csv not found - run train.py first.")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    test = pd.read_csv(split_path).query("split == 'test'")

    picks = {}
    for i, (cls, g) in enumerate(sorted(test.groupby("class"))):
        row = g.sample(1, random_state=SEED).iloc[0]
        picks[cls] = Image.open(ROOT / row.path).convert("RGB")
        save(picks[cls], f"{i + 1:02d}_{CLASS_INFO[cls]['display'].replace(' / ', '_').replace(' ', '_')}")

    degraded = [
        ("plastic", {"gamma": 0.3}, "Plastic_dim_lighting"),
        ("metal", {"glare": 0.9}, "Metal_harsh_glare"),
        ("plastic", {"occlusion_pct": 50}, "Plastic_hand_occluded_50pct"),
        ("biological", {"blur": 8}, "Food_motion_blur"),
    ]
    for j, (cls, kw, name) in enumerate(degraded):
        save(apply_perturbations(picks[cls], **kw), f"{20 + j}_{name}")

    ood_dir = ROOT / "data" / "imagenette2-160" / "val"
    if ood_dir.exists():
        rng = random.Random(SEED)
        # n02102040 = English springer (dog), n03028079 = church, n03888257 = parachute
        for k, (wnid, label) in enumerate([("n02102040", "dog"), ("n03028079", "church"),
                                           ("n03888257", "parachute")]):
            files = sorted((ood_dir / wnid).glob("*.JPEG"))
            if files:
                save(Image.open(rng.choice(files)), f"{30 + k}_OUT_OF_SCOPE_{label}")
    rng = np.random.default_rng(SEED)
    save(Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)), "40_OUT_OF_SCOPE_noise")

    # real-world photos outside every dataset (product / everyday photos)
    real = {"kitkat": "Real_chocolate_wrapper", "cake": "Real_leftover_cake", "purple_bottle": "Real_water_bottle"}
    for k, (src, name) in enumerate(real.items()):
        p = ROOT / "data" / "user_tests" / f"{src}.jpg"
        if p.exists():
            save(Image.open(p), f"{50 + k}_{name}")

    print(f"Wrote {len(list(OUT.glob('*.jpg')))} images to {OUT}")


if __name__ == "__main__":
    main()

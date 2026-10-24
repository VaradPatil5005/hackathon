"""
train.py
Documented stratified 70/15/15 split + transfer-learning fine-tuning of an
ImageNet-pretrained ConvNeXt-Tiny on the 12-class Garbage Classification
dataset. The test split is never used here; evaluate.py uses it once.

Usage:
    python train.py --data_dir data/garbage_classification --epochs 5
Outputs (in artifacts/):
    split.csv            every image path with its split (train/val/test)
    split_summary.json   per-class counts for each split
    model.pt             best checkpoint by validation macro-F1
    training_log.json    per-epoch loss / accuracy / macro-F1
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from waste_core import (
    ARTIFACTS,
    CLASS_INFO,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_PATH,
    ROOT,
    build_model,
    eval_transform,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform(image_size: int = 224):
    """Lighting (ColorJitter) and occlusion (RandomErasing) augmentation for robustness."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.35, saturation=0.3, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), value="random"),
        ]
    )


class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(ROOT / self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[i]


def make_split(data_dir: Path) -> pd.DataFrame:
    """Stratified 70 / 15 / 15 split with a fixed seed, saved to artifacts/split.csv."""
    class_names = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    unknown = [c for c in class_names if c not in CLASS_INFO]
    if unknown:
        raise SystemExit(f"Unknown class folders {unknown}; expected {sorted(CLASS_INFO)}")

    rows = []
    for label, cls in enumerate(class_names):
        for p in sorted((data_dir / cls).iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": p.relative_to(ROOT).as_posix(), "class": cls, "label": label})
    df = pd.DataFrame(rows)

    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=df.loc[temp_idx, "label"], random_state=SEED
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    df["category"] = df["class"].map(lambda c: CLASS_INFO[c]["category"])

    ARTIFACTS.mkdir(exist_ok=True)
    df.to_csv(ARTIFACTS / "split.csv", index=False)

    counts = df.pivot_table(index="class", columns="split", values="path", aggfunc="count").fillna(0).astype(int)
    counts = counts[["train", "val", "test"]]
    counts["total"] = counts.sum(axis=1)
    summary = {
        "seed": SEED,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "total_images": int(len(df)),
        "class_names": class_names,
        "per_class": {
            cls: {**{k: int(v) for k, v in counts.loc[cls].items()}, "category": CLASS_INFO[cls]["category"]}
            for cls in class_names
        },
        "split_totals": {s: int((df["split"] == s).sum()) for s in ["train", "val", "test"]},
    }
    with open(ARTIFACTS / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(counts.to_string())
    print("Split totals:", summary["split_totals"])
    return df


def run_epoch_eval(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(x.to(device, non_blocking=True))
            preds.append(out.argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return float((preds == labels).mean()), float(f1_score(labels, preds, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/garbage_classification")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    seed_everything()
    data_dir = (ROOT / args.data_dir).resolve()
    df = make_split(data_dir)
    class_names = sorted(df["class"].unique())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    tr, va = df[df.split == "train"], df[df.split == "val"]
    train_loader = DataLoader(
        ImageListDataset(tr.path, tr.label, train_transform()),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        ImageListDataset(va.path, va.label, eval_transform()),
        batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True,
    )

    model = build_model(len(class_names)).to(device)

    # Class-balanced loss: 'clothes' has ~5x more images than most classes.
    counts = np.bincount(tr.label, minlength=len(class_names))
    weights = torch.tensor((counts.sum() / (len(counts) * counts)) ** 0.5, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    log, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / seen:.4f}", flush=True)

        val_acc, val_f1 = run_epoch_eval(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": running / seen,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "seconds": round(time.time() - t0, 1),
        }
        log.append(entry)
        print(f"Epoch {epoch}/{args.epochs} | loss {entry['train_loss']:.4f} | val acc {val_acc:.4f} | "
              f"val macro-F1 {val_f1:.4f} | {entry['seconds']}s", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "meta": {
                        "architecture": "ConvNeXt-Tiny (ImageNet-1k pretrained, fine-tuned)",
                        "image_size": 224,
                        "best_epoch": epoch,
                        "val_accuracy": val_acc,
                        "val_macro_f1": val_f1,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                    },
                },
                MODEL_PATH,
            )
            print(f"  saved best checkpoint -> {MODEL_PATH.name}")

    with open(ARTIFACTS / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Done. Best val macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()

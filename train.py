"""Train a tangent-keypoint regressor.

Reads all per-video labels from dataset/labels/*.csv. Each row points to a
frame image under dataset/images/<video>/frame_<idx>.png and gives the two
tangent endpoints in original-frame pixel coordinates. The model predicts the
4 normalized coords (tx1, ty1, tx2, ty2) in [0,1].

Re-run this whenever new labels arrive. The label tool will hot-reload the
checkpoint when you press R.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
LABEL_DIR = DATASET / "labels"
IMG_DIR = DATASET / "images"
MODELS = ROOT / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_samples() -> list:
    """Return list of (img_path, [tx1,ty1,tx2,ty2] normalized) tuples."""
    samples = []
    if not LABEL_DIR.exists():
        return samples
    for csv_path in sorted(LABEL_DIR.glob("*.csv")):
        video_stem = csv_path.stem
        images_root = IMG_DIR / video_stem
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                frame = int(row["frame"])
                img_path = images_root / f"frame_{frame:06d}.png"
                if not img_path.exists():
                    continue
                with Image.open(img_path) as im:
                    w, h = im.size
                coords = [
                    float(row["tx1"]) / w,
                    float(row["ty1"]) / h,
                    float(row["tx2"]) / w,
                    float(row["ty2"]) / h,
                ]
                samples.append((img_path, coords))
    return samples


class TangentDataset(Dataset):
    def __init__(self, samples, train: bool):
        self.samples = samples
        self.train = train
        steps = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if train:
            steps.insert(1, transforms.ColorJitter(brightness=0.15, contrast=0.15))
        self.tf = transforms.Compose(steps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, coords = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(coords, dtype=torch.float32)
        return x, y


def build_model() -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 4)
    return m


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    crit = nn.SmoothL1Loss(reduction="sum")
    total_loss, total_px_err, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = crit(pred, y)
            if train:
                optimizer.zero_grad()
                (loss / x.size(0)).backward()
                optimizer.step()
            total_loss += loss.item()
            # mean per-point L1 in normalized units (multiply by image dim for px)
            total_px_err += (pred - y).abs().mean(dim=1).sum().item()
            n += x.size(0)
    return total_loss / max(n, 1), total_px_err / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    MODELS.mkdir(parents=True, exist_ok=True)

    samples = collect_samples()
    if len(samples) < 4:
        print(f"Only {len(samples)} labeled samples found — label more in label_tool.py first.")
        return
    print(f"Loaded {len(samples)} labeled frames across {len(set(s[0].parent for s in samples))} videos.")

    n_val = max(1, int(round(args.val_frac * len(samples))))
    n_train = len(samples) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(samples, [n_train, n_val], generator=g)
    train_ds = TangentDataset(list(train_subset), train=True)
    val_ds = TangentDataset(list(val_subset), train=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_err = float("inf")
    best_path = MODELS / "tangent_best.pt"
    last_path = MODELS / "tangent_last.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_err = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_err = run_epoch(model, val_loader, device)
        scheduler.step()
        print(f"epoch {epoch:03d} | train_loss {tr_loss:.5f} err {tr_err:.4f} "
              f"| val_loss {va_loss:.5f} err {va_err:.4f}")
        ckpt = {"state_dict": model.state_dict(), "epoch": epoch, "val_err": va_err}
        torch.save(ckpt, last_path)
        if va_err < best_err:
            best_err = va_err
            torch.save(ckpt, best_path)

    print(f"Best val normalized-coord L1: {best_err:.4f}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()

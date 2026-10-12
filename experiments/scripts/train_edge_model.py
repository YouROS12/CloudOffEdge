"""
Train a lightweight edge model on PlantWild.

Default architecture: MobileNetV3-Small (ImageNet-pretrained), fine-tuned end-to-end.
Saves a torchvision-style checkpoint plus a sidecar metadata JSON.

Design choices (lifted from paper-1 / PlantEdge what was already validated):
  - Reuse PlantEdge's MVPDR-exact PlantWild split via paper1.src.datasets.load_plantwild
  - ImageNet normalization, 224x224 input (MobileNetV3 native)
  - AdamW + cosine schedule + linear warmup
  - FocalLoss(gamma=2, label_smoothing=0.1)
  - Sqrt-frequency class-balanced sampler (PlantEdge convention)
  - AMP mixed precision
  - Carve a 10% val split out of train for best-checkpoint selection

Outputs:
  outputs/edge_model/best.pt          - checkpoint (state_dict + config + class_names)
  outputs/edge_model/history.json     - per-epoch loss/val_acc/val_f1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


def _add_turboquant_root_to_path() -> Path:
    """Make experiments/scripts importable (PlantWild split loader); returns the repository root."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return Path(__file__).resolve().parents[2]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class _ItemsDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(label)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p = torch.exp(-ce)
        return ((1 - p) ** self.gamma * ce).mean()


def stratified_train_val_split(
    items: List[Tuple[str, int]], val_frac: float = 0.1, seed: int = 1,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Per-class shuffle so val gets a slice of every class."""
    import random
    by_cls = {}
    for path, lbl in items:
        by_cls.setdefault(lbl, []).append((path, lbl))
    rng = random.Random(seed)
    train, val = [], []
    for lbl, lst in by_cls.items():
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * val_frac))
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_balanced_sampler(items: List[Tuple[str, int]]) -> WeightedRandomSampler:
    """Sqrt-frequency class weighting (PlantEdge convention)."""
    counts = {}
    for _, lbl in items:
        counts[lbl] = counts.get(lbl, 0) + 1
    cls_weight = {c: 1.0 / math.sqrt(n) for c, n in counts.items()}
    weights = [cls_weight[lbl] for _, lbl in items]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(items),
        replacement=True,
    )


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def build_model(arch: str, n_classes: int) -> nn.Module:
    if arch == "mobilenet_v3_small":
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, n_classes)
        return m
    if arch == "mobilenet_v3_large":
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, n_classes)
        return m
    if arch == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, n_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        y_pred.extend(logits.argmax(-1).cpu().tolist())
        y_true.extend(y.tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-dir", default="outputs/edge_model")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    root = _add_turboquant_root_to_path()
    print(f"[paths] TurboQuant root = {root}")

    from plantwild import load_plantwild

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    train_items_full, test_items, class_names = load_plantwild()
    n_classes = len(class_names)
    print(f"[data] train={len(train_items_full)}  test={len(test_items)}  n_classes={n_classes}")

    train_items, val_items = stratified_train_val_split(
        train_items_full, val_frac=0.1, seed=args.seed,
    )
    print(f"[split] train={len(train_items)}  val={len(val_items)}")

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = _ItemsDataset(train_items, train_tf)
    val_ds = _ItemsDataset(val_items, eval_tf)
    test_ds = _ItemsDataset(test_items, eval_tf)

    sampler = make_balanced_sampler(train_items)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(args.arch, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.arch}  params={n_params/1e6:.2f}M")

    loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(total_steps * args.warmup_frac))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    history_path = out_dir / "history.json"

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    t0 = time.time()
    step = 0

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_b = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr_mult = cosine_with_warmup(step, total_steps, warmup)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_mult

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item()
            n_b += 1
            step += 1

        val_m = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": ep_loss / max(1, n_b),
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        print(f"  ep{epoch:3d}  loss={row['train_loss']:.4f}  "
              f"val_acc={val_m['accuracy']*100:.2f}%  val_f1={val_m['macro_f1']*100:.2f}%  "
              f"({row['elapsed_s']:.0f}s)")

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "n_classes": n_classes,
                "class_names": class_names,
                "img_size": args.img_size,
                "epoch": epoch,
                "val_metrics": val_m,
            }, ckpt_path)
            history_path.write_text(json.dumps(history, indent=2))

    # Restore best for final test eval
    print(f"[best] epoch={best_epoch}  val_f1={best_val_f1*100:.2f}%")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    test_m = evaluate(model, test_loader, device)
    print(f"[test] acc={test_m['accuracy']*100:.2f}%  f1={test_m['macro_f1']*100:.2f}%")

    state["test_metrics"] = test_m
    torch.save(state, ckpt_path)
    history.append({"epoch": best_epoch, "test_metrics": test_m})
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[done] saved -> {ckpt_path}")


if __name__ == "__main__":
    main()

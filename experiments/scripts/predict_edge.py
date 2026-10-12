"""
Run a trained edge model on the PlantWild test split and write a
predictions CSV consumable by experiments/scripts/run_experiment.py.

Output columns: image_path, true_label, edge_pred, edge_confidence

We use the same PlantEdge loader as train_edge_model.py + export_paper1_predictions.py
so the test ordering and class_names line up with cloud_predictions_test.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
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


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _ItemsDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(label), path


def build_model(arch: str, n_classes: int):
    factories = {
        "mobilenet_v3_small": mobilenet_v3_small,
        "mobilenet_v3_large": mobilenet_v3_large,
        "efficientnet_b0":    efficientnet_b0,
    }
    if arch not in factories:
        raise ValueError(f"Unknown arch: {arch}")
    m = factories[arch](weights=None)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = torch.nn.Linear(in_feats, n_classes)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="outputs/edge_model/best.pt")
    ap.add_argument("--out-csv", default="outputs/edge_predictions_test.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    root = _add_turboquant_root_to_path()
    print(f"[paths] TurboQuant root = {root}")

    from plantwild import load_plantwild

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = state.get("arch", "mobilenet_v3_small")
    n_classes = int(state["n_classes"])
    class_names = list(state["class_names"])
    img_size = int(state.get("img_size", 224))
    print(f"[ckpt] arch={arch}  n_classes={n_classes}  img_size={img_size}  "
          f"epoch={state.get('epoch', '?')}")

    model = build_model(arch, n_classes).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    _, test_items, loader_class_names = load_plantwild()
    if loader_class_names != class_names:
        raise SystemExit(
            "FATAL: class_names mismatch between checkpoint and current loader. "
            "The PlantWild split may have changed."
        )
    print(f"[data] test={len(test_items)}")

    eval_tf = build_eval_transform(img_size)
    ds = _ItemsDataset(test_items, eval_tf)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    rows = []
    all_logits = []
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            top2 = probs.topk(2, dim=-1)
            confs = top2.values[:, 0]
            preds = top2.indices[:, 0]
            margins = top2.values[:, 0] - top2.values[:, 1]
            # Entropy in nats; clamp for numerical stability
            entropies = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
            all_logits.append(logits.cpu())
            for path, true_id, pid, conf, marg, ent in zip(
                paths,
                y.tolist(),
                preds.cpu().tolist(),
                confs.cpu().tolist(),
                margins.cpu().tolist(),
                entropies.cpu().tolist(),
            ):
                rows.append((
                    path,
                    class_names[true_id],
                    class_names[pid],
                    f"{conf:.6f}",
                    f"{marg:.6f}",
                    f"{ent:.6f}",
                ))

    correct = sum(1 for _, t, p, _, _, _ in rows if t == p)
    print(f"[eval] top-1 accuracy = {correct/len(rows)*100:.2f}%  ({correct}/{len(rows)})")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_path", "true_label", "edge_pred",
            "edge_confidence", "edge_margin", "edge_entropy",
        ])
        w.writerows(rows)
    print(f"[write] {out_csv}  ({len(rows)} rows)")

    # Dump full logits for downstream learned-router training
    logits_path = out_csv.with_suffix(".logits.pt")
    torch.save({
        "image_paths": [r[0] for r in rows],
        "true_labels": [r[1] for r in rows],
        "logits": torch.cat(all_logits, dim=0),
        "class_names": class_names,
    }, logits_path)
    print(f"[write] {logits_path}  shape={tuple(torch.cat(all_logits, dim=0).shape)}")


if __name__ == "__main__":
    main()

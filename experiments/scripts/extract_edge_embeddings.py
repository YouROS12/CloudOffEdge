"""
Extract penultimate-layer embeddings from a trained edge classifier
on PlantWild val + test splits, for use as the "edge -> cloud" message
in the cheap-embedding verification pipeline (Option B).

For MobileNetV3 (small/large) and EfficientNet-B0, the penultimate
representation is the output of `features` -> avgpool -> classifier[:-1].
We grab it via a forward hook on the second-to-last classifier sub-module.

Output (saved as .pt):
  {
    "image_paths": [...],
    "true_labels": [...]              # int class ids
    "edge_pred":   tensor int64 (N,),
    "edge_logits": tensor float32 (N, C),
    "edge_emb":    tensor float32 (N, D),
    "class_names": [...],
    "arch":        str,
    "emb_dim":     int,
  }

Bandwidth note: at fp16 the message size is 2*D bytes/sample
  (v3-Large -> 1280*2 = 2.5 KB, v3-Small -> 1024*2 = 2 KB,
   effnet-b0 -> 1280*2 = 2.5 KB).
"""

from __future__ import annotations

import argparse
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


def install_penultimate_hook(model: torch.nn.Module, arch: str):
    """Hook the input to the final classifier Linear so we capture the
    actual penultimate representation the head sees. Returns a list
    that gets populated per-batch."""
    captured: List[torch.Tensor] = []
    head = model.classifier[-1]

    def _hook(module, inputs, output):
        # `inputs` is a tuple; for Linear it's (x,)
        captured.append(inputs[0].detach().to("cpu", torch.float32))

    handle = head.register_forward_hook(_hook)
    return captured, handle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Path to edge model best.pt")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True, help="Output .pt path")
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
    print(f"[ckpt] arch={arch}  n_classes={n_classes}  img_size={img_size}")

    model = build_model(arch, n_classes).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Decide which split to walk.
    train_full, test_items, loader_class_names = load_plantwild()
    if loader_class_names != class_names:
        raise SystemExit("FATAL: class_names mismatch with current loader.")

    if args.split == "test":
        items = test_items
    else:
        # Use PlantEdge's PlantWild val split (1820 samples) so we line up
        # with paper-2's cloud_predictions_val.csv.
        from plantwild import PlantWild
        pw = PlantWild()
        if list(pw.classnames) != class_names:
            raise SystemExit("FATAL: PlantEdge PlantWild classnames disagree with ckpt.")
        items = [(d.impath, int(d.label)) for d in pw.val]

    print(f"[data] split={args.split}  n={len(items)}")

    eval_tf = build_eval_transform(img_size)
    ds = _ItemsDataset(items, eval_tf)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    captured, handle = install_penultimate_hook(model, arch)

    image_paths: List[str] = []
    true_labels: List[int] = []
    all_logits: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)  # hook fills `captured`
            all_logits.append(logits.detach().to("cpu", torch.float32))
            image_paths.extend(list(paths))
            true_labels.extend([int(v) for v in y.tolist()])

    handle.remove()

    edge_logits = torch.cat(all_logits, dim=0)
    edge_emb = torch.cat(captured, dim=0)
    edge_pred = edge_logits.argmax(dim=-1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "image_paths": image_paths,
        "true_labels": torch.tensor(true_labels, dtype=torch.long),
        "edge_pred":   edge_pred,
        "edge_logits": edge_logits,
        "edge_emb":    edge_emb,
        "class_names": class_names,
        "arch":        arch,
        "emb_dim":     int(edge_emb.shape[1]),
    }, out_path)

    acc = (edge_pred == torch.tensor(true_labels)).float().mean().item()
    print(f"[write] {out_path}  emb_shape={tuple(edge_emb.shape)}  "
          f"logits_shape={tuple(edge_logits.shape)}  "
          f"top1={acc*100:.2f}%")
    print(f"[bandwidth] fp16 message = {edge_emb.shape[1]*2/1024:.2f} KB / sample "
          f"(emb only); + ~{n_classes*2/1024:.2f} KB if logits also sent")


if __name__ == "__main__":
    main()

"""
Run the edge model on the PlantWild VAL split (1820 unseen samples)
and dump logits + cloud predictions, so we can train a learned router
on val and evaluate on the full test set (no i.i.d. leak).

Outputs:
  outputs/edge_predictions_val.csv     - same schema as test edge CSV
  outputs/edge_predictions_val.logits.pt
  outputs/cloud_predictions_val.csv    - cloud predictions on val

Cloud predictions for val: we re-use the trained QualityGate head
on val features (cached + concatenated + L2-normalised the same way
as for test).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


def _add_root():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_tf(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _ItemsDataset(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), int(label), path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/edge_model/best.pt")
    ap.add_argument("--out-csv", default="outputs/edge_predictions_val.csv")
    ap.add_argument("--edge-only", action="store_true",
                    help="Skip the (already cached) cloud predictions on val.")
    args = ap.parse_args()

    root = _add_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reuse PlantEdge's PlantWild loader to get the val split with image paths
    from plantwild import PlantWild
    pw = PlantWild()
    val_items = [(d.impath, d.label) for d in pw.val]
    class_names = list(pw.classnames)
    print(f"[loader] val={len(val_items)}  n_classes={len(class_names)}")

    # ── 1) Edge predictions on val ──────────────────────────────────────
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if list(state["class_names"]) != class_names:
        raise SystemExit("FATAL: edge ckpt class_names disagree with loader")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from predict_edge import build_model
    model = build_model(state.get("arch", "mobilenet_v3_small"), len(class_names))
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    tf = eval_tf(int(state.get("img_size", 224)))
    loader = DataLoader(_ItemsDataset(val_items, tf), batch_size=64,
                        num_workers=2, pin_memory=(device == "cuda"))

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
            entropies = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
            all_logits.append(logits.cpu())
            for path, t, p, c, m, e in zip(
                paths, y.tolist(), preds.cpu().tolist(),
                confs.cpu().tolist(), margins.cpu().tolist(), entropies.cpu().tolist(),
            ):
                rows.append((path, class_names[t], class_names[p],
                             f"{c:.6f}", f"{m:.6f}", f"{e:.6f}"))

    edge_csv = Path(args.out_csv)
    edge_csv.parent.mkdir(parents=True, exist_ok=True)
    with edge_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_label", "edge_pred",
                    "edge_confidence", "edge_margin", "edge_entropy"])
        w.writerows(rows)
    print(f"[edge] acc = "
          f"{sum(1 for r in rows if r[1] == r[2]) / len(rows) * 100:.2f}%  "
          f"-> {edge_csv}")

    logits_path = edge_csv.with_suffix(".logits.pt")
    torch.save({
        "image_paths": [r[0] for r in rows],
        "true_labels": [r[1] for r in rows],
        "logits": torch.cat(all_logits, dim=0),
        "class_names": class_names,
    }, logits_path)
    print(f"[edge] {logits_path}  shape={tuple(torch.cat(all_logits, dim=0).shape)}")
    if args.edge_only:
        return

    # ── 2) Cloud predictions on val ─────────────────────────────────────
    # Pull val items by re-extracting features for each backbone is expensive;
    # but the paper-1 caches DON'T contain val features (only train+test).
    # Cheapest path: extract DINOv2/DINOv3/CLIP features for val fresh.
    # Since this is only 1820 images x 3 backbones, do it once and cache.

    print("\n[cloud] extracting val features for the 3 backbones (one-time)")
    from paper1.src.backbones import load_backbone

    val_paths = [r[0] for r in rows]
    val_labels_idx = torch.tensor([class_names.index(r[1]) for r in rows], dtype=torch.long)

    backbones = ["dinov2_vitl14", "dinov3_vitl16", "clip_vitl14"]
    cache_dir = Path("outputs/val_features_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    fused = []
    for bname in backbones:
        cache_p = cache_dir / f"{bname}.pt"
        if cache_p.exists():
            print(f"  [cache] {bname}")
            d = torch.load(cache_p, map_location="cpu", weights_only=False)
        else:
            print(f"  [extract] {bname}")
            extract_fn, feat_dim, preprocess, img_size = load_backbone(bname, device)
            ds = _ItemsDataset(val_items, preprocess)
            ld = DataLoader(ds, batch_size=32, num_workers=2,
                            pin_memory=(device == "cuda"))
            feats, paths_seen = [], []
            with torch.no_grad():
                for x, _, paths_b in ld:
                    x = x.to(device, non_blocking=True)
                    feats.append(extract_fn(x).cpu().float())
                    paths_seen.extend(paths_b)
            feats = torch.cat(feats, dim=0)
            d = {"features": feats, "image_paths": paths_seen}
            torch.save(d, cache_p)
        # Align to val_paths order
        if d["image_paths"] != val_paths:
            idx_map = {p: i for i, p in enumerate(d["image_paths"])}
            order = [idx_map[p] for p in val_paths]
            d["features"] = d["features"][order]
        fused.append(d["features"])

    fused_feats = torch.cat(fused, dim=-1)
    fused_feats = F.normalize(fused_feats, dim=-1)
    print(f"[cloud] fused shape = {tuple(fused_feats.shape)}")

    # Load trained QualityGate head
    head_ckpt = torch.load("outputs/cloud_head/qualitygate_fusion_3L.pt",
                           map_location=device, weights_only=False)
    from paper1.src.classifiers import build_classifier
    n_classes = len(class_names)
    head = build_classifier("qualitygate", fused_feats.shape[-1], n_classes,
                            n_prototypes=8).to(device)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()

    with torch.no_grad():
        logits = head(fused_feats.to(device))
        probs = F.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)
    cloud_acc = (preds.cpu() == val_labels_idx).float().mean().item()
    print(f"[cloud] val acc = {cloud_acc * 100:.2f}%")

    cloud_csv = Path("outputs/cloud_predictions_val.csv")
    with cloud_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_label", "cloud_pred", "cloud_confidence"])
        for path, true_name, p, c in zip(val_paths,
                                         [class_names[i] for i in val_labels_idx.tolist()],
                                         preds.cpu().tolist(), confs.cpu().tolist()):
            w.writerow([path, true_name, class_names[p], f"{c:.6f}"])
    print(f"[cloud] -> {cloud_csv}")


if __name__ == "__main__":
    main()

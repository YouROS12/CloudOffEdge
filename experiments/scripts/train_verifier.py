"""
Train a small cloud-side verifier MLP that decides, given the edge's
penultimate embedding + edge logits, whether the sample should be
escalated to the full cloud pipeline.

Supervision signal:
  target = 1  iff  cloud is likely to *correct* the edge.
We try two label variants:
  - "useful_offload":   y=1 iff edge_pred != true AND cloud_pred == true
  - "cloud_disagrees":  y=1 iff edge_pred != cloud_pred  (label-free fallback)
Use of true labels here is OK because supervision happens on the val
split (cloud-side training data), and the verifier is then evaluated
on the held-out test split.

Inputs:
  --emb-pt    : edge embeddings/logits .pt produced by extract_edge_embeddings.py
  --cloud-csv : cloud predictions CSV (image_path, cloud_pred, true_label)
  --out       : output checkpoint path

Output checkpoint contains:
  state_dict, input_dim, hidden_dim, label_mode, val_metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class VerifierMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_features(emb: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Concatenate edge embedding + a few cheap logit-derived signals.

    We feed the verifier:
      - edge_emb                         (D dims)
      - softmax probs                    (C dims)
      - confidence, margin, entropy      (3 scalars)
    """
    probs = F.softmax(logits, dim=-1)
    top2 = probs.topk(2, dim=-1)
    conf = top2.values[:, 0:1]
    margin = (top2.values[:, 0:1] - top2.values[:, 1:2])
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
    return torch.cat([emb, probs, conf, margin, entropy], dim=-1)


def merge_with_cloud(blob: dict, cloud_csv: Path) -> dict:
    cloud = pd.read_csv(cloud_csv)
    df_blob = pd.DataFrame({
        "image_path": blob["image_paths"],
        "_idx": np.arange(len(blob["image_paths"])),
    })
    merged = df_blob.merge(cloud, on="image_path", how="inner",
                           suffixes=("", "_cloud"))
    if len(merged) != len(df_blob):
        raise SystemExit(
            f"Cloud merge dropped rows: {len(df_blob)} -> {len(merged)}. "
            f"Image-path mismatch between edge and cloud CSVs."
        )
    # Re-order to match merged ordering
    order = merged["_idx"].to_numpy()
    blob_o = {
        "image_paths": [blob["image_paths"][i] for i in order],
        "true_labels": blob["true_labels"][order],
        "edge_pred":   blob["edge_pred"][order],
        "edge_logits": blob["edge_logits"][order],
        "edge_emb":    blob["edge_emb"][order],
        "class_names": blob["class_names"],
    }
    # Build cloud_pred / true_label arrays as ints aligned with blob_o
    name_to_id = {n: i for i, n in enumerate(blob_o["class_names"])}
    cloud_pred_id = merged["cloud_pred"].map(name_to_id).to_numpy()
    true_id = merged["true_label"].map(name_to_id).to_numpy()
    blob_o["cloud_pred"] = torch.tensor(cloud_pred_id, dtype=torch.long)
    blob_o["true_id"] = torch.tensor(true_id, dtype=torch.long)
    # Sanity: the true_id from cloud csv should match blob true_labels
    if not torch.equal(blob_o["true_id"], blob_o["true_labels"]):
        n_match = (blob_o["true_id"] == blob_o["true_labels"]).sum().item()
        print(f"[warn] true_label mismatch in {len(blob_o['true_id'])-n_match} rows "
              f"(using edge-side true_labels)")
    return blob_o


def make_targets(blob: dict, mode: str) -> torch.Tensor:
    edge = blob["edge_pred"]
    cloud = blob["cloud_pred"]
    true = blob["true_labels"]
    if mode == "useful_offload":
        y = ((edge != true) & (cloud == true)).long()
    elif mode == "cloud_disagrees":
        y = (edge != cloud).long()
    elif mode == "cloud_correct":
        # broader: route when cloud is more likely right than edge
        y = (cloud == true).long()
    else:
        raise ValueError(f"Unknown label mode: {mode}")
    return y


def train_one_split(X: torch.Tensor, y: torch.Tensor, val_frac: float,
                    epochs: int, batch_size: int, lr: float, device: str,
                    hidden_dim: int, seed: int = 1):
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    pos = ytr.sum().item()
    neg = len(ytr) - pos
    pos_weight = torch.tensor([max(1.0, neg / max(1, pos))], device=device)
    print(f"[split] tr={len(Xtr)} (pos={pos}, neg={neg})  va={len(Xva)}  "
          f"pos_weight={pos_weight.item():.2f}")

    model = VerifierMLP(X.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_d = Xtr.to(device); ytr_d = ytr.to(device).float()
    Xva_d = Xva.to(device); yva_d = yva.to(device).float()

    ds = TensorDataset(Xtr_d, ytr_d)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_auc = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(Xva_d).cpu().numpy()
        try:
            auc = roc_auc_score(yva_d.cpu().numpy(), vlogits)
        except ValueError:
            auc = float("nan")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d}  val_auc={auc:.4f}  best={best_auc:.4f}")

    model.load_state_dict(best_state)
    return model, best_auc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-pt", required=True)
    ap.add_argument("--cloud-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-mode", default="useful_offload",
                    choices=["useful_offload", "cloud_disagrees", "cloud_correct"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    blob = torch.load(args.emb_pt, map_location="cpu", weights_only=False)
    print(f"[emb] arch={blob.get('arch')}  N={len(blob['image_paths'])}  "
          f"emb_dim={blob['edge_emb'].shape[1]}  "
          f"n_classes={len(blob['class_names'])}")

    blob = merge_with_cloud(blob, Path(args.cloud_csv))

    X = build_features(blob["edge_emb"], blob["edge_logits"])
    y = make_targets(blob, args.label_mode)
    print(f"[features] X={tuple(X.shape)}  y_pos_rate={y.float().mean().item():.3f}  "
          f"label_mode={args.label_mode}")

    model, best_auc = train_one_split(
        X, y, args.val_frac, args.epochs, args.batch_size, args.lr,
        device, args.hidden_dim, args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(X.shape[1]),
        "emb_dim": int(blob["edge_emb"].shape[1]),
        "n_classes": len(blob["class_names"]),
        "hidden_dim": args.hidden_dim,
        "label_mode": args.label_mode,
        "best_val_auc": float(best_auc),
        "class_names": blob["class_names"],
        "arch": blob.get("arch"),
    }, out)
    print(f"[done] saved -> {out}  best_val_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()

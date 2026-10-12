"""
Apply the trained cloud-side verifier to the test split, sweep its
threshold, and produce a bandwidth-vs-accuracy curve to compare
against the heuristic and learned-router baselines.

Cost model
----------
  Per sample, edge ALWAYS sends the embedding + logits over the wire:
      msg_kb = 2 * (emb_dim + n_classes) / 1024     # fp16
  When the verifier escalates (rate r), the edge additionally
  uploads the full image:
      image_kb = 250.0  (configurable via --image-kb)
  So average bandwidth per sample = msg_kb + r * image_kb.

  Latency:
      always: edge_lat + msg_send_lat + verifier_lat
      escalate: + image_send_lat + cloud_lat + return_lat

Accuracy
--------
  At threshold t:
      escalate = score > t
      final_pred = where(escalate, cloud_pred, edge_pred)

We also report the same heuristic baselines (confidence/margin/entropy)
on the SAME embedding blob for a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


def load_test_blob(emb_pt: Path, cloud_csv: Path) -> dict:
    blob = torch.load(emb_pt, map_location="cpu", weights_only=False)
    cloud = pd.read_csv(cloud_csv)
    df = pd.DataFrame({
        "image_path": blob["image_paths"],
        "_idx": np.arange(len(blob["image_paths"])),
    }).merge(cloud, on="image_path", how="inner")
    if len(df) != len(blob["image_paths"]):
        raise SystemExit(f"cloud merge dropped rows {len(blob['image_paths'])} -> {len(df)}")
    order = df["_idx"].to_numpy()
    name_to_id = {n: i for i, n in enumerate(blob["class_names"])}
    cloud_pred_id = df["cloud_pred"].map(name_to_id).to_numpy()
    return {
        "image_paths": [blob["image_paths"][i] for i in order],
        "true_labels": blob["true_labels"][order],
        "edge_pred":   blob["edge_pred"][order],
        "edge_logits": blob["edge_logits"][order],
        "edge_emb":    blob["edge_emb"][order],
        "cloud_pred":  torch.tensor(cloud_pred_id, dtype=torch.long),
        "class_names": blob["class_names"],
        "arch":        blob.get("arch"),
    }


def build_features(emb, logits):
    probs = F.softmax(logits, dim=-1)
    top2 = probs.topk(2, dim=-1)
    conf = top2.values[:, 0:1]
    margin = top2.values[:, 0:1] - top2.values[:, 1:2]
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
    return torch.cat([emb, probs, conf, margin, entropy], dim=-1)


def sweep(score, edge_pred, cloud_pred, true,
          higher_is_uncertain=True, n_thresh=120,
          msg_kb=4.0, image_kb=250.0,
          edge_lat=50.0, verifier_lat=2.0,
          cloud_lat=250.0, send_lat=80.0, ret_lat=20.0):
    edge_pred = np.asarray(edge_pred); cloud_pred = np.asarray(cloud_pred)
    true = np.asarray(true); score = np.asarray(score)
    rows = []
    grid = np.linspace(score.min(), score.max(), n_thresh)
    for t in grid:
        offload = (score > t) if higher_is_uncertain else (score < t)
        final = np.where(offload, cloud_pred, edge_pred)
        rate = float(offload.mean())
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(true, final)),
            "offload_rate": rate,
            "avg_bandwidth_kb": msg_kb + rate * image_kb,
            "avg_latency_ms": (edge_lat + verifier_lat
                               + rate * (send_lat + cloud_lat + ret_lat)),
        })
    return pd.DataFrame(rows)


def at_offload(c: pd.DataFrame, t: float) -> pd.Series:
    return c.loc[(c["offload_rate"] - t).abs().idxmin()]


def at_bandwidth(c: pd.DataFrame, kb: float) -> pd.Series:
    return c.loc[(c["avg_bandwidth_kb"] - kb).abs().idxmin()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-pt", required=True, help="Test-split embeddings .pt")
    ap.add_argument("--cloud-csv", required=True)
    ap.add_argument("--verifier", required=True, help="Trained verifier .pt")
    ap.add_argument("--out-dir", default="results/option_B_verifier")
    ap.add_argument("--image-kb", type=float, default=250.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    blob = load_test_blob(Path(args.emb_pt), Path(args.cloud_csv))
    n_classes = len(blob["class_names"])
    emb_dim = blob["edge_emb"].shape[1]

    msg_bytes_fp16 = 2 * (emb_dim + n_classes)
    msg_kb = msg_bytes_fp16 / 1024.0
    print(f"[bandwidth] always-on edge->cloud msg = {msg_kb:.3f} KB / sample "
          f"(emb {emb_dim}-d + logits {n_classes}-d at fp16)")

    # Heuristic baselines on the same data
    probs = F.softmax(blob["edge_logits"], dim=-1)
    top2 = probs.topk(2, dim=-1)
    conf = top2.values[:, 0].numpy()
    margin = (top2.values[:, 0] - top2.values[:, 1]).numpy()
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1).numpy()
    edge_pred = blob["edge_pred"].numpy()
    cloud_pred = blob["cloud_pred"].numpy()
    true = blob["true_labels"].numpy()

    edge_acc = float((edge_pred == true).mean())
    cloud_acc = float((cloud_pred == true).mean())
    target = 0.95 * cloud_acc

    print(f"[baseline] edge={edge_acc*100:.2f}%  cloud={cloud_acc*100:.2f}%  "
          f"target(95%)={target*100:.2f}%")

    # Verifier scoring
    from train_verifier import VerifierMLP  # type: ignore
    ck = torch.load(args.verifier, map_location=device, weights_only=False)
    if ck["emb_dim"] != emb_dim or ck["n_classes"] != n_classes:
        raise SystemExit(f"verifier dim mismatch: ck={ck['emb_dim']}+{ck['n_classes']}  "
                         f"data={emb_dim}+{n_classes}")
    model = VerifierMLP(ck["input_dim"], hidden_dim=ck["hidden_dim"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    X = build_features(blob["edge_emb"], blob["edge_logits"]).to(device)
    with torch.no_grad():
        verifier_score = torch.sigmoid(model(X)).cpu().numpy()

    # AUCs (vs. useful_offload labels)
    yu = ((edge_pred != true) & (cloud_pred == true)).astype(int)
    yd = (edge_pred != cloud_pred).astype(int)
    print(f"[AUC vs useful_offload]  verifier={roc_auc_score(yu, verifier_score):.4f}  "
          f"1-conf={roc_auc_score(yu, 1-conf):.4f}  "
          f"1-margin={roc_auc_score(yu, 1-margin):.4f}  "
          f"entropy={roc_auc_score(yu, entropy):.4f}")
    print(f"[AUC vs cloud_disagrees] verifier={roc_auc_score(yd, verifier_score):.4f}")

    curves = {
        "verifier":   sweep(verifier_score, edge_pred, cloud_pred, true,
                            higher_is_uncertain=True, msg_kb=msg_kb, image_kb=args.image_kb),
        "confidence": sweep(conf, edge_pred, cloud_pred, true,
                            higher_is_uncertain=False, msg_kb=msg_kb, image_kb=args.image_kb),
        "margin":     sweep(margin, edge_pred, cloud_pred, true,
                            higher_is_uncertain=False, msg_kb=msg_kb, image_kb=args.image_kb),
        "entropy":    sweep(entropy, edge_pred, cloud_pred, true,
                            higher_is_uncertain=True, msg_kb=msg_kb, image_kb=args.image_kb),
    }

    # Oracle bound (knows useful_offload exactly)
    benefit = (cloud_pred == true).astype(int) - (edge_pred == true).astype(int)
    order = np.argsort(-benefit)
    cum = np.zeros(len(true), dtype=bool)
    oracle_rows = []
    for k in range(len(true) + 1):
        if k > 0:
            cum[order[k - 1]] = True
        rate = k / len(true)
        final = np.where(cum, cloud_pred, edge_pred)
        oracle_rows.append({"k": k, "offload_rate": rate,
                            "accuracy": float(accuracy_score(true, final)),
                            "avg_bandwidth_kb": msg_kb + rate * args.image_kb})
    oracle = pd.DataFrame(oracle_rows)

    # Tables: at fixed offload + at fixed bandwidth
    print("\nAt offload budgets:")
    print(f"  {'budget':<8} {'verifier':>10} {'confidence':>10} {'margin':>10} {'entropy':>10} {'oracle':>10}")
    for b in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        row = f"  {b*100:5.0f}%   "
        for n in ["verifier", "confidence", "margin", "entropy"]:
            r = at_offload(curves[n], b)
            row += f" {r['accuracy']*100:9.2f}%"
        ridx = (oracle["offload_rate"] - b).abs().idxmin()
        row += f" {oracle.loc[ridx, 'accuracy']*100:9.2f}%"
        print(row)

    print("\nAt avg bandwidth budgets (KB/sample):")
    print(f"  {'budget':<8} {'verifier':>10} {'confidence':>10} {'margin':>10} {'entropy':>10} {'oracle':>10}")
    for kb in [10, 20, 40, 60, 80, 100]:
        row = f"  {kb:>5} KB "
        for n in ["verifier", "confidence", "margin", "entropy"]:
            r = at_bandwidth(curves[n], kb)
            row += f" {r['accuracy']*100:9.2f}%"
        ridx = (oracle["avg_bandwidth_kb"] - kb).abs().idxmin()
        row += f" {oracle.loc[ridx, 'accuracy']*100:9.2f}%"
        print(row)

    print("\nGO/NO-GO (95% of cloud, <=30% offload, this script's image_kb):")
    for n, c in curves.items():
        hits = c[c["accuracy"] >= target].sort_values("offload_rate")
        if not len(hits):
            print(f"  {n:<12} NEVER")
            continue
        r = hits.iloc[0]
        verdict = "GO " if r["offload_rate"] <= 0.30 else "NO-GO"
        print(f"  {n:<12} earliest offload={r['offload_rate']*100:5.2f}%  "
              f"acc={r['accuracy']*100:5.2f}%  "
              f"bw={r['avg_bandwidth_kb']:6.2f} KB  -> {verdict}")
    h = oracle[oracle["accuracy"] >= target]
    if len(h):
        r = h.iloc[0]
        print(f"  {'oracle':<12} earliest offload={r['offload_rate']*100:5.2f}%  "
              f"acc={r['accuracy']*100:5.2f}%  bw={r['avg_bandwidth_kb']:6.2f} KB")

    # Save curves
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for n, c in curves.items():
        c2 = c.copy(); c2["signal"] = n; all_rows.append(c2)
    oc = oracle.copy(); oc["signal"] = "oracle"; oc["threshold"] = np.nan
    oc["avg_latency_ms"] = np.nan
    all_rows.append(oc[["threshold", "accuracy", "offload_rate",
                        "avg_bandwidth_kb", "avg_latency_ms", "signal"]])
    pd.concat(all_rows, ignore_index=True).to_csv(
        out_dir / "verifier_sweep.csv", index=False)

    summary = {
        "edge_acc": edge_acc, "cloud_acc": cloud_acc, "target_95": target,
        "msg_kb_always": msg_kb, "image_kb": args.image_kb,
        "verifier_best_val_auc": ck.get("best_val_auc"),
        "auc_vs_useful_offload": {
            "verifier": float(roc_auc_score(yu, verifier_score)),
            "1-conf": float(roc_auc_score(yu, 1-conf)),
            "1-margin": float(roc_auc_score(yu, 1-margin)),
            "entropy": float(roc_auc_score(yu, entropy)),
        },
    }
    (out_dir / "verifier_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[write] {out_dir/'verifier_sweep.csv'}")
    print(f"[write] {out_dir/'verifier_summary.json'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        # (1) accuracy vs offload rate
        ax = axes[0]
        for n, c in curves.items():
            c = c.sort_values("offload_rate")
            ls = "-" if n == "verifier" else "--"
            lw = 2.0 if n == "verifier" else 1.0
            ax.plot(c["offload_rate"]*100, c["accuracy"]*100, ls=ls, lw=lw,
                    marker="." if n == "verifier" else None, label=n)
        ax.plot(oracle["offload_rate"]*100, oracle["accuracy"]*100,
                color="black", lw=1.5, label="oracle")
        ax.axhline(cloud_acc*100, ls="--", color="grey", lw=1, label="cloud-only")
        ax.axhline(target*100, ls="-.", color="red", lw=1, label="95% target")
        ax.axvline(30, ls=":", color="green", lw=1, label="30% offload")
        ax.set_xlabel("Offload rate (%)"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Verifier vs heuristics — accuracy vs offload")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

        # (2) accuracy vs avg bandwidth
        ax = axes[1]
        for n, c in curves.items():
            c = c.sort_values("avg_bandwidth_kb")
            ls = "-" if n == "verifier" else "--"
            lw = 2.0 if n == "verifier" else 1.0
            ax.plot(c["avg_bandwidth_kb"], c["accuracy"]*100, ls=ls, lw=lw,
                    marker="." if n == "verifier" else None, label=n)
        ax.plot(oracle["avg_bandwidth_kb"], oracle["accuracy"]*100,
                color="black", lw=1.5, label="oracle")
        ax.axhline(cloud_acc*100, ls="--", color="grey", lw=1, label="cloud-only")
        ax.axhline(target*100, ls="-.", color="red", lw=1, label="95% target")
        # always-image baseline reference
        ax.scatter([msg_kb + args.image_kb], [cloud_acc*100], marker="*",
                   s=120, color="purple", label="always-image (cloud-only)")
        ax.scatter([msg_kb], [edge_acc*100], marker="X",
                   s=80, color="orange", label="msg-only (no escalation)")
        ax.set_xlabel("Avg bandwidth per sample (KB)"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Verifier vs heuristics — accuracy vs bandwidth")
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(out_dir / "verifier_curves.png", dpi=150)
        print(f"[plot] {out_dir/'verifier_curves.png'}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()

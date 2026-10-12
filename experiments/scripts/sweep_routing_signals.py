"""
Compare three edge-routing signals: confidence, margin (top1-top2), entropy.

For confidence and margin: HIGH = trust the edge -> threshold says "offload if signal < t".
For entropy:               HIGH = uncertain -> we instead "offload if signal > t".

We compute the same accuracy / offload-rate / latency tradeoff curve for each
signal and write a unified CSV + a comparison plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def merge_csvs(edge_csv: str, cloud_csv: str) -> pd.DataFrame:
    edge = pd.read_csv(edge_csv)
    cloud = pd.read_csv(cloud_csv)
    return edge.merge(cloud, on="image_path", how="inner", suffixes=("", "_cloud"))


def sweep(df: pd.DataFrame, signal_col: str, *, higher_is_uncertain: bool,
          n_thresholds: int = 80, edge_lat: float = 50.0,
          cloud_lat: float = 250.0, send: float = 100.0, ret: float = 30.0,
          image_kb: float = 250.0) -> pd.DataFrame:
    sig = df[signal_col].to_numpy()
    lo, hi = float(sig.min()), float(sig.max())
    thresholds = np.linspace(lo, hi, n_thresholds)
    rows = []
    edge_pred = df["edge_pred"].to_numpy()
    cloud_pred = df["cloud_pred"].to_numpy()
    true = df["true_label"].to_numpy()
    for t in thresholds:
        if higher_is_uncertain:
            offload = sig > t
        else:
            offload = sig < t
        final = np.where(offload, cloud_pred, edge_pred)
        rows.append({
            "signal": signal_col,
            "threshold": float(t),
            "accuracy": float(accuracy_score(true, final)),
            "macro_f1": float(f1_score(true, final, average="macro", zero_division=0)),
            "offload_rate": float(offload.mean()),
            "average_latency_ms": float(np.where(offload, edge_lat + send + cloud_lat + ret, edge_lat).mean()),
            "bandwidth_kb_per_sample": float(offload.mean() * image_kb),
            "num_samples": int(len(df)),
        })
    return pd.DataFrame(rows)


def at_offload(curve: pd.DataFrame, target: float) -> pd.Series:
    idx = (curve["offload_rate"] - target).abs().idxmin()
    return curve.loc[idx]


def main() -> None:
    edge_csv = "outputs/edge_predictions_test.csv"
    cloud_csv = "outputs/cloud_predictions_test.csv"
    out_dir = Path("results/phase0_routing_signals")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_csvs(edge_csv, cloud_csv)
    print(f"[data] merged rows = {len(merged)}")

    cloud_acc = float((merged["cloud_pred"] == merged["true_label"]).mean())
    edge_acc = float((merged["edge_pred"] == merged["true_label"]).mean())
    target_acc = 0.95 * cloud_acc

    sweeps = {
        "confidence": sweep(merged, "edge_confidence", higher_is_uncertain=False),
        "margin":     sweep(merged, "edge_margin",     higher_is_uncertain=False),
        "entropy":    sweep(merged, "edge_entropy",    higher_is_uncertain=True),
    }

    all_rows = pd.concat(sweeps.values(), ignore_index=True)
    all_rows.to_csv(out_dir / "sweep_all_signals.csv", index=False)

    # Comparison table at fixed offload budgets
    print("\n" + "=" * 78)
    print(f"Edge-only acc = {edge_acc*100:.2f}%   Cloud-only acc = {cloud_acc*100:.2f}%   "
          f"95% target = {target_acc*100:.2f}%")
    print("=" * 78)
    for budget in [0.10, 0.20, 0.30, 0.40, 0.50]:
        print(f"\nAt offload \u2248 {budget*100:.0f}%:")
        print(f"  {'signal':<12} {'acc':>8} {'%cloud':>8} {'lat_ms':>9} {'offload':>9}")
        for name, curve in sweeps.items():
            r = at_offload(curve, budget)
            print(f"  {name:<12} {r['accuracy']*100:7.2f}% {r['accuracy']/cloud_acc*100:7.2f}% "
                  f"{r['average_latency_ms']:8.1f}  {r['offload_rate']*100:7.2f}%")

    # GO/NO-GO per signal
    print("\n" + "=" * 78)
    print("GO/NO-GO check (95% of cloud at \u226430% offload)")
    print("=" * 78)
    for name, curve in sweeps.items():
        hits = curve[curve["accuracy"] >= target_acc]
        if len(hits) == 0:
            print(f"  {name:<12} target NEVER reached  (best={curve['accuracy'].max()*100:.2f}%)")
            continue
        r = hits.sort_values("offload_rate").iloc[0]
        verdict = "GO " if r["offload_rate"] <= 0.30 else "NO-GO"
        print(f"  {name:<12} earliest hit: offload={r['offload_rate']*100:5.2f}%  "
              f"acc={r['accuracy']*100:.2f}%  thr={r['threshold']:.4f}  -> {verdict}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        for name, curve in sweeps.items():
            curve = curve.sort_values("offload_rate")
            plt.plot(curve["offload_rate"] * 100, curve["accuracy"] * 100,
                     marker=".", label=name, linewidth=1.5)
        plt.axhline(cloud_acc * 100, ls="--", color="grey", label="cloud-only")
        plt.axhline(edge_acc * 100, ls=":", color="grey", label="edge-only")
        plt.axhline(target_acc * 100, ls="-.", color="red", label="95% of cloud target")
        plt.axvline(30, ls="-.", color="green", label="30% offload budget")
        plt.xlabel("Offload rate (%)")
        plt.ylabel("Accuracy (%)")
        plt.title("Selective edge-cloud: routing-signal comparison")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "signals_comparison.png", dpi=150)
        print(f"\n[plot] {out_dir / 'signals_comparison.png'}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()

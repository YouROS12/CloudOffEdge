"""
Oracle-routing upper bound: at each offload rate, route the K samples
where the edge is wrong / cloud is right, in priority order.

This tells us the *best possible* accuracy curve a signal-based router
could ever reach for a given (edge, cloud) pair. If the oracle still
misses the GO criterion, the edge model is the bottleneck and no
amount of router-engineering will help.

Two oracle policies:
  oracle_edge_wrong:  offload exactly the samples where edge_pred != true_label
  oracle_cloud_right: offload samples where cloud helps
                      (edge wrong AND cloud right -> "useful offload")

We sort the offload candidates by (cloud_right - edge_right) so the most
useful offloads happen first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def main() -> None:
    edge = pd.read_csv("outputs/edge_predictions_test.csv")
    cloud = pd.read_csv("outputs/cloud_predictions_test.csv")
    df = edge.merge(cloud, on="image_path", how="inner", suffixes=("", "_cloud"))
    print(f"[data] merged = {len(df)}")

    edge_right = (df["edge_pred"] == df["true_label"]).to_numpy()
    cloud_right = (df["cloud_pred"] == df["true_label"]).to_numpy()
    true = df["true_label"].to_numpy()
    edge_pred = df["edge_pred"].to_numpy()
    cloud_pred = df["cloud_pred"].to_numpy()
    n = len(df)

    edge_acc = float(edge_right.mean())
    cloud_acc = float(cloud_right.mean())
    target = 0.95 * cloud_acc

    # Useful offload = edge wrong AND cloud right (a "win")
    # Wasted offload = edge right AND cloud wrong (a "loss")
    # Net benefit per sample if offloaded:
    #   +1 if edge wrong & cloud right
    #    0 if both wrong or both right
    #   -1 if edge right & cloud wrong
    benefit = cloud_right.astype(int) - edge_right.astype(int)

    # Sort indices by descending benefit (best offload candidates first).
    order = np.argsort(-benefit)

    rows = []
    cum_offload = np.zeros(n, dtype=bool)
    for k in range(n + 1):
        if k > 0:
            cum_offload[order[k - 1]] = True
        final = np.where(cum_offload, cloud_pred, edge_pred)
        rows.append({
            "k_offload": k,
            "offload_rate": k / n,
            "accuracy": float(accuracy_score(true, final)),
        })
    curve = pd.DataFrame(rows)
    out_dir = Path("results/phase0_routing_signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / "oracle_curve.csv", index=False)

    print(f"\nEdge-only={edge_acc*100:.2f}%   Cloud-only={cloud_acc*100:.2f}%   "
          f"95% target={target*100:.2f}%")
    print(f"\n{'offload':>10} {'oracle_acc':>12} {'%cloud':>10}")
    for tgt in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        idx = (curve["offload_rate"] - tgt).abs().idxmin()
        r = curve.loc[idx]
        print(f"  {r['offload_rate']*100:7.1f}%  {r['accuracy']*100:9.2f}%  "
              f"{r['accuracy']/cloud_acc*100:7.1f}%")

    hits = curve[curve["accuracy"] >= target]
    if len(hits) == 0:
        print("\nORACLE never reaches 95% target  -> edge model is too weak (need Option 2)")
    else:
        first = hits.iloc[0]
        verdict = "GO " if first["offload_rate"] <= 0.30 else "NO-GO"
        print(f"\nORACLE reaches target at offload={first['offload_rate']*100:.2f}%  -> {verdict}")
        if first["offload_rate"] > 0.30:
            print("    Even the oracle needs >30% offload to hit 95% of cloud accuracy.")
            print("    The edge model is the bottleneck.  Recommended next step: Option 2 (stronger edge).")
        else:
            print("    Oracle CAN hit it inside 30% offload.")
            print("    Signal engineering (calibration, learned router) might close the gap.")

    # Plot oracle vs the three signal curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sigs = pd.read_csv(out_dir / "sweep_all_signals.csv")
        plt.figure(figsize=(7, 5))
        plt.plot(curve["offload_rate"] * 100, curve["accuracy"] * 100,
                 label="oracle (upper bound)", color="black", linewidth=2)
        for name in ["confidence", "margin", "entropy"]:
            sub = sigs[sigs["signal"] == name].sort_values("offload_rate")
            plt.plot(sub["offload_rate"] * 100, sub["accuracy"] * 100,
                     marker=".", label=name, linewidth=1.2)
        plt.axhline(cloud_acc * 100, ls="--", color="grey", label="cloud-only")
        plt.axhline(edge_acc * 100, ls=":", color="grey", label="edge-only")
        plt.axhline(target * 100, ls="-.", color="red", label="95% of cloud")
        plt.axvline(30, ls="-.", color="green", label="30% offload budget")
        plt.xlabel("Offload rate (%)")
        plt.ylabel("Accuracy (%)")
        plt.title("Oracle vs signal-based routing")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "oracle_vs_signals.png", dpi=150)
        print(f"[plot] {out_dir / 'oracle_vs_signals.png'}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()

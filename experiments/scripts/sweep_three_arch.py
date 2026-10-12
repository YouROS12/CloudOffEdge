"""
3-way edge architecture comparison on PlantWild routing:
  - MobileNetV3-Small  (Phase 0 baseline)
  - MobileNetV3-Large  (Option A winner)
  - EfficientNet-B0    (Option A.3 candidate)

Reuses the analyze() helper from sweep_v3l.py to keep the same
threshold sweep / GO/NO-GO definitions, then plots all three on
one accuracy-vs-offload chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from sweep_v3l import analyze  # type: ignore


CONFIGS = [
    ("v3-Small",       "outputs/edge_predictions_test.csv"),
    ("v3-Large",       "outputs/edge_v3l_predictions_test.csv"),
    ("EfficientNet-B0","outputs/edge_effb0_predictions_test.csv"),
]


def main() -> None:
    out_dir = Path("results/option_A_3arch")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 78)
    print("# OPTION A.3: edge-architecture sweep on PlantWild routing")
    print("#" * 78)

    results = {}
    for label, csv_path in CONFIGS:
        if not Path(csv_path).exists():
            print(f"[skip] {label}: {csv_path} not found")
            continue
        curves, oracle, edge_acc, cloud_acc = analyze(csv_path, label)
        results[label] = {
            "curves": curves, "oracle": oracle,
            "edge_acc": edge_acc, "cloud_acc": cloud_acc,
        }

    # Persist combined CSV
    rows = []
    for label, d in results.items():
        for sig, c in d["curves"].items():
            cc = c.copy()
            cc["signal"] = sig
            cc["arch"] = label
            rows.append(cc)
        oc = d["oracle"].copy()
        oc["signal"] = "oracle"
        oc["arch"] = label
        oc["threshold"] = pd.NA
        oc["bandwidth_kb_per_sample"] = pd.NA
        oc["average_latency_ms"] = pd.NA
        rows.append(oc)
    pd.concat(rows, ignore_index=True).to_csv(
        out_dir / "three_arch_sweep.csv", index=False)
    print(f"\n[write] {out_dir/'three_arch_sweep.csv'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))
        # Pick 'margin' as the canonical heuristic for clarity, plus oracle.
        cloud_acc = next(iter(results.values()))["cloud_acc"]
        target = 0.95 * cloud_acc

        colors = {"v3-Small": "tab:gray",
                  "v3-Large": "tab:blue",
                  "EfficientNet-B0": "tab:orange"}
        for label, d in results.items():
            color = colors.get(label, "black")
            for sig in ("confidence", "margin", "entropy"):
                if sig in d["curves"]:
                    cc = d["curves"][sig].sort_values("offload_rate")
                    plt.plot(cc["offload_rate"]*100, cc["accuracy"]*100,
                             color=color, lw=1.5, alpha=0.85,
                             label=f"{label} {sig}" if sig == "margin" else None,
                             ls={"confidence": ":", "margin": "-", "entropy": "--"}[sig])
            o = d["oracle"].sort_values("offload_rate")
            plt.plot(o["offload_rate"]*100, o["accuracy"]*100,
                     color=color, lw=2.5, ls="-",
                     label=f"{label} oracle", alpha=0.4)

        plt.axhline(cloud_acc*100, ls="--", color="black", lw=1, label="cloud-only")
        plt.axhline(target*100, ls="-.", color="red", lw=1, label="95% target")
        plt.axvline(30, ls=":", color="green", lw=1, label="30% budget")
        plt.xlabel("Offload rate (%)"); plt.ylabel("Accuracy (%)")
        plt.title("PlantWild routing: edge architecture comparison")
        plt.legend(fontsize=8, loc="lower right", ncols=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "three_arch_offload.png", dpi=150)
        print(f"[plot] {out_dir/'three_arch_offload.png'}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()

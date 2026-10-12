"""Sweep v3-Large edge predictions vs cloud, compare against v3-Small baseline."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def sweep(scores, edge_pred, cloud_pred, true,
          higher_is_uncertain=True, n_thresh=80,
          edge_lat=50.0, cloud_lat=250.0, send=100.0, ret=30.0, image_kb=250.0):
    rows = []
    for t in np.linspace(scores.min(), scores.max(), n_thresh):
        offload = (scores > t) if higher_is_uncertain else (scores < t)
        final = np.where(offload, cloud_pred, edge_pred)
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(true, final)),
            "offload_rate": float(offload.mean()),
            "average_latency_ms": float(np.where(offload, edge_lat + send + cloud_lat + ret, edge_lat).mean()),
            "bandwidth_kb_per_sample": float(offload.mean() * image_kb),
        })
    return pd.DataFrame(rows)


def at_offload(c, t):
    return c.loc[(c["offload_rate"] - t).abs().idxmin()]


def analyze(edge_csv, label, cloud_csv="outputs/cloud_predictions_test.csv"):
    edge = pd.read_csv(edge_csv)
    cloud = pd.read_csv(cloud_csv)
    df = edge.merge(cloud, on="image_path", how="inner", suffixes=("", "_cloud"))
    edge_pred = df["edge_pred"].to_numpy()
    cloud_pred = df["cloud_pred"].to_numpy()
    true = df["true_label"].to_numpy()
    edge_right = (edge_pred == true)
    cloud_right = (cloud_pred == true)
    edge_acc = float(edge_right.mean())
    cloud_acc = float(cloud_right.mean())
    target = 0.95 * cloud_acc

    conf = df["edge_confidence"].to_numpy()
    margin = df["edge_margin"].to_numpy()
    entropy = df["edge_entropy"].to_numpy()

    curves = {
        "confidence": sweep(conf, edge_pred, cloud_pred, true, higher_is_uncertain=False),
        "margin":     sweep(margin, edge_pred, cloud_pred, true, higher_is_uncertain=False),
        "entropy":    sweep(entropy, edge_pred, cloud_pred, true, higher_is_uncertain=True),
    }

    benefit = cloud_right.astype(int) - edge_right.astype(int)
    order = np.argsort(-benefit)
    oracle_rows = []
    cum = np.zeros(len(true), dtype=bool)
    for k in range(len(true) + 1):
        if k > 0:
            cum[order[k - 1]] = True
        final = np.where(cum, cloud_pred, edge_pred)
        oracle_rows.append({"k": k, "offload_rate": k / len(true),
                            "accuracy": float(accuracy_score(true, final))})
    oracle = pd.DataFrame(oracle_rows)

    print(f"\n{'='*78}\n[{label}]  edge={edge_acc*100:.2f}%  cloud={cloud_acc*100:.2f}%  target={target*100:.2f}%\n{'='*78}")
    for budget in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        print(f"\nAt offload \u2248 {budget*100:.0f}%:")
        print(f"  {'signal':<14} {'acc':>7} {'%cloud':>7}")
        for n, c in curves.items():
            r = at_offload(c, budget)
            print(f"  {n:<14} {r['accuracy']*100:6.2f}% {r['accuracy']/cloud_acc*100:6.2f}%")
        idx = (oracle["offload_rate"] - budget).abs().idxmin()
        r = oracle.loc[idx]
        print(f"  {'oracle':<14} {r['accuracy']*100:6.2f}% {r['accuracy']/cloud_acc*100:6.2f}%")

    print(f"\nGO/NO-GO ({label}):")
    for n, c in curves.items():
        hits = c[c["accuracy"] >= target]
        if not len(hits):
            print(f"  {n:<14} NEVER")
            continue
        r = hits.sort_values("offload_rate").iloc[0]
        verdict = "GO " if r["offload_rate"] <= 0.30 else "NO-GO"
        print(f"  {n:<14} earliest offload={r['offload_rate']*100:5.2f}%  acc={r['accuracy']*100:.2f}%  -> {verdict}")
    h = oracle[oracle["accuracy"] >= target]
    if len(h):
        r = h.iloc[0]
        print(f"  {'oracle':<14} earliest offload={r['offload_rate']*100:5.2f}%  acc={r['accuracy']*100:.2f}%")

    return curves, oracle, edge_acc, cloud_acc


def main():
    out = Path("results/option_A_v3l")
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 78)
    print("# OPTION A: stronger edge model")
    print("#" * 78)

    cs, ors, ea_s, ca = analyze("outputs/edge_predictions_test.csv", "v3-Small (baseline)")
    cl, orl, ea_l, _ = analyze("outputs/edge_v3l_predictions_test.csv", "v3-Large")

    # Save curves
    for n, c in cs.items():
        c["signal"] = n
        c["arch"] = "v3_small"
    for n, c in cl.items():
        c["signal"] = n
        c["arch"] = "v3_large"
    pd.concat(list(cs.values()) + list(cl.values()), ignore_index=True).to_csv(
        out / "v3small_vs_v3large_sweep.csv", index=False)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        target = 0.95 * ca
        plt.figure(figsize=(8, 5.5))
        plt.plot(ors["offload_rate"]*100, ors["accuracy"]*100, color="black", ls="--",
                 label="oracle v3-S", linewidth=1.5)
        plt.plot(orl["offload_rate"]*100, orl["accuracy"]*100, color="black",
                 label="oracle v3-L", linewidth=2)
        for n, c in cs.items():
            c = c.sort_values("offload_rate")
            plt.plot(c["offload_rate"]*100, c["accuracy"]*100, ls=":", marker=".",
                     label=f"v3-S {n}", linewidth=1)
        for n, c in cl.items():
            c = c.sort_values("offload_rate")
            plt.plot(c["offload_rate"]*100, c["accuracy"]*100, ls="-", marker=".",
                     label=f"v3-L {n}", linewidth=1.4)
        plt.axhline(ca*100, ls="--", color="grey", label="cloud-only")
        plt.axhline(target*100, ls="-.", color="red", label="95% target")
        plt.axvline(30, ls="-.", color="green", label="30% offload budget")
        plt.xlabel("Offload rate (%)"); plt.ylabel("Accuracy (%)")
        plt.title("Stronger edge model (v3-Large vs v3-Small) on PlantWild test")
        plt.legend(loc="lower right", fontsize=7, ncols=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "v3small_vs_v3large.png", dpi=150)
        print(f"\n[plot] {out / 'v3small_vs_v3large.png'}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()

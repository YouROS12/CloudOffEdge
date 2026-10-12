"""Validation-calibrated escalation threshold from per-image prediction CSVs.

Rule (Section 3.4): tau is the smallest margin threshold whose validation
accuracy (escalate iff margin < tau) reaches 95 % of the cloud's validation
accuracy; it is then applied unchanged to the test split.

  python experiments/scripts/calibrate_threshold.py --edge-val outputs/edge_v3l_predictions_val.csv \
      --edge-test outputs/edge_v3l_predictions_test.csv --out results/threshold_calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "artifacts"


def key(r):
    return r["true_label"] + "/" + r["image_path"].replace("\\", "/").rsplit("/", 1)[-1]


def load(edge_csv: Path, cloud_csv: Path):
    cloud = {key(r): r["cloud_pred"] for r in csv.DictReader(cloud_csv.open(encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(edge_csv.open(encoding="utf-8")):
        rows.append((float(r["edge_margin"]), r["edge_pred"] == r["true_label"], cloud[key(r)] == r["true_label"]))
    return rows


def stats(rows, tau):
    n = len(rows)
    esc = [m < tau for m, _, _ in rows]
    acc = sum(c if e else ed for e, (_, ed, c) in zip(esc, rows)) / n
    cloud = sum(c for _, _, c in rows) / n
    return {"n": n, "cloud_acc": cloud, "edge_acc": sum(ed for _, ed, _ in rows) / n,
            "escalation": sum(esc) / n, "acc": acc, "pct_of_cloud": acc / cloud,
            "reduction_x": n / max(1, sum(esc))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-val", required=True)
    ap.add_argument("--edge-test", required=True)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    val = load(Path(args.edge_val), OUT / "cloud_predictions_val.csv")
    test = load(Path(args.edge_test), OUT / "cloud_predictions_test.csv")
    goal = args.target * sum(c for _, _, c in val) / len(val)
    cands = sorted({m for m, _, _ in val})
    # escalate iff margin < tau: thresholds just above each observed margin
    tau = next(round(m + 1e-6, 6) for m in [0.0] + cands if stats(val, m + 1e-6)["acc"] >= goal)
    res = {"rule": "smallest margin threshold whose validation accuracy reaches 95% of cloud validation "
                   "accuracy; escalate iff margin < tau",
           "edge_val_csv": args.edge_val, "tau": tau, "val": stats(val, tau), "test": stats(test, tau)}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

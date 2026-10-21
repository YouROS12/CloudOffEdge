"""Validation-calibrated operating points of the original and distilled edge models.

Same inputs and rule as the deployed model: the fp32 per-image prediction CSVs
written by experiments/scripts/predict_edge.py, and calibrate_threshold.py's rule (tau is the
smallest margin threshold whose validation accuracy reaches 95 % of the cloud's
validation accuracy; applied unchanged to the test split). The original model
therefore reproduces Table 7 exactly, and the distilled model the deployed tau.

  python experiments/scripts/eval_kd_operating_point.py -> results/edge_loop/kd_operating_point.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_threshold import OUT, load, stats  # noqa: E402

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
MODELS = {"baseline": "edge_v3l", "distilled": "edge_v3l_kd"}


def calibrate(val, target=0.95):
    goal = target * sum(c for _, _, c in val) / len(val)
    return next(round(m + 1e-6, 6) for m in [0.0] + sorted({m for m, _, _ in val})
                if stats(val, m + 1e-6)["acc"] >= goal)


def esc_for(rows, goal_acc):
    """Smallest escalation rate (least-confident first) whose accuracy reaches goal_acc."""
    rows = sorted(rows, key=lambda r: r[0])
    n, acc = len(rows), sum(ed for _, ed, _ in rows)
    if acc / n >= goal_acc:
        return 0.0
    for k, (_, ed, c) in enumerate(rows, 1):
        acc += c - ed
        if acc / n >= goal_acc:
            return k / n
    return None


def main() -> None:
    report = {}
    for name, stem in MODELS.items():
        val = load(OUT / f"{stem}_predictions_val.csv", OUT / "cloud_predictions_val.csv")
        test = load(OUT / f"{stem}_predictions_test.csv", OUT / "cloud_predictions_test.csv")
        tau = calibrate(val)
        if name == "baseline":
            # the deployed tau (results/threshold_calibration.json, 0.138424); recalibrating from the
            # CSVs' 6-decimal margins gives 0.138339, which differs on one test image (683 vs 684)
            report["baseline_recalibrated_tau"] = tau
            tau = json.loads((PAPER2 / "results" / "threshold_calibration.json").read_text())["tau"]
        v, t = stats(val, tau), stats(test, tau)
        report[name] = {"tau": tau, "val_escalation": v["escalation"], "test_edge_acc": t["edge_acc"],
                        "test_escalation": t["escalation"], "test_acc": t["acc"],
                        "test_pct_of_cloud": t["pct_of_cloud"], "test_reduction_x": t["reduction_x"],
                        "test_esc_for_95": esc_for(test, 0.95 * t["cloud_acc"])}
        print(name, {k: round(x, 5) for k, x in report[name].items()})
    out = PAPER2 / "results" / "edge_loop" / "kd_operating_point.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

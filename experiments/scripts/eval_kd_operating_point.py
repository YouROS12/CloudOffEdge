"""Validation-calibrated operating point of the distilled edge model.

Same rule as the deployed model (results/threshold_calibration.json): tau is
the smallest margin threshold whose validation accuracy reaches 95 % of the
cloud's validation accuracy; it is then applied unchanged to the test split.

  python experiments/scripts/eval_kd_operating_point.py -> results/edge_loop/kd_operating_point.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_edge_model as base  # noqa: E402
from train_edge_distill import OUT, load_split, margin, operating, predict  # noqa: E402

PAPER2 = Path(__file__).resolve().parents[2]  # repository root


def main() -> None:
    _, test_items, names, soft = load_split()
    idx = {n: i for i, n in enumerate(names)}
    with (OUT / "cloud_predictions_val.csv").open(encoding="utf-8") as fh:
        val = list(csv.DictReader(fh))
    val_items = [(r["image_path"], idx[r["true_label"]]) for r in val]
    yv = torch.tensor([l for _, l in val_items])
    cv = torch.tensor([idx[r["cloud_pred"]] for r in val])
    yt, ct = soft["test_labels"], torch.tensor([idx[r["cloud_pred"]] for r in
                                                 csv.DictReader((OUT / "cloud_predictions_test.csv").open(encoding="utf-8"))])

    _, eval_tf = base.build_transforms(224)
    report = {}
    for name, ck in {"baseline": OUT / "edge_model_v3l" / "best.pt", "distilled": OUT / "edge_model_v3l_kd" / "best.pt"}.items():
        m = base.build_model("mobilenet_v3_large", len(names))
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
        m = m.cuda()
        lv, lt = predict(m, val_items, eval_tf, "cuda"), predict(m, test_items, eval_tf, "cuda")
        mv, mt = margin(lv), margin(lt)
        goal = 0.95 * (cv == yv).float().mean().item()
        tau = None
        for t in torch.sort(torch.unique(mv)).values.tolist() + [1.01]:
            acc = torch.where(mv < t, cv == yv, lv.argmax(-1) == yv).float().mean().item()
            if acc >= goal:
                tau = t
                break
        esc = mt < tau
        fin = torch.where(esc, ct == yt, lt.argmax(-1) == yt)
        report[name] = {"tau": tau, "val_escalation": (mv < tau).float().mean().item(),
                        "test_edge_acc": (lt.argmax(-1) == yt).float().mean().item(),
                        "test_escalation": esc.float().mean().item(), "test_acc": fin.float().mean().item(),
                        "test_pct_of_cloud": fin.float().mean().item() / (ct == yt).float().mean().item(),
                        "test_reduction_x": 1 / esc.float().mean().item(),
                        "test_esc_for_95": operating(lt, ct, yt)["esc_for_95"]}
        print(name, {k: round(v, 4) for k, v in report[name].items()})
    out = PAPER2 / "results" / "edge_loop" / "kd_operating_point.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

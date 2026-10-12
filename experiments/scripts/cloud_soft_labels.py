"""Cloud-ensemble soft labels for the train and test splits, from cached features.

Paper 1 cached the three frozen backbones' global descriptors for PlantWild
(train 13 045 / test 3 677, same seed-fixed split). Applying the QualityGate
head to them gives the ensemble's class probabilities for every image, the
teacher signal for distillation. Test-split predictions are checked against
outputs/cloud_predictions_test.csv (the paper's offline cloud predictions):
same accuracy and same per-image labels => same order and same head.

  python experiments/scripts/cloud_soft_labels.py -> outputs/cloud_soft_labels.pt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
sys.path[:0] = [str(PAPER2 / "rasp_deploy"), str(PAPER2 / "jetson_deploy")]
from cloud.server import QualityGate  # noqa: E402

CACHE = Path(__import__("os").environ.get("PAPER1_CACHED_FEATURES",
                                         Path(__file__).resolve().parents[2] / "data" / "paper1_cached_features"))
OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
BACKBONES = ["dinov2_vitl14", "dinov3_vitl16", "clip_vitl14"]


def main() -> None:
    caches = [torch.load(CACHE / f"plantwild__{b}.pt", map_location="cpu", weights_only=False) for b in BACKBONES]
    names = caches[0]["class_names"]
    ck = torch.load(OUT_DIR / "cloud_head" / "qualitygate_fusion_3L.pt", map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    head = QualityGate(sd["linear.weight"].shape[1], len(names),
                       n_prototypes=ck.get("result", {}).get("n_prototypes", 8))
    head.load_state_dict(sd, strict=True)
    head.eval()
    out = {"class_names": names}
    for split in ("train", "test"):
        for c in caches[1:]:
            assert torch.equal(c[f"{split}_labels"], caches[0][f"{split}_labels"]), "label order differs"
        x = F.normalize(torch.cat([F.normalize(c[f"{split}_features"].float(), dim=-1) for c in caches], -1), dim=-1)
        with torch.no_grad():
            logits = head(x)
        y = caches[0][f"{split}_labels"]
        out[f"{split}_logits"], out[f"{split}_labels"] = logits, y
        print(f"{split}: n={len(y)}, ensemble top-1 {(logits.argmax(1) == y).float().mean():.4f}")

    with (OUT_DIR / "cloud_predictions_test.csv").open(encoding="utf-8") as fh:
        ref = [r["cloud_pred"] for r in csv.DictReader(fh)]
    mine = [names[i] for i in out["test_logits"].argmax(1).tolist()]
    agree = sum(a == b for a, b in zip(mine, ref)) / len(ref)
    print(f"test: per-image agreement with cloud_predictions_test.csv (same order) = {agree:.4f}")
    torch.save(out, OUT_DIR / "cloud_soft_labels.pt")
    print("wrote", OUT_DIR / "cloud_soft_labels.pt")


if __name__ == "__main__":
    main()

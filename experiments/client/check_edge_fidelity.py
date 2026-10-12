"""Edge fidelity check: does the deployed edge node reproduce the offline
edge predictions?

Sends N test images (original file bytes, as a camera would) to an edge
node with force_route="edge" and compares the returned label and margin
with the offline predictions CSV (experiments/scripts/predict_edge.py output).

  python check_edge_fidelity.py --image-dir <plantwild_test> \
      --offline outputs/edge_v3l_predictions_test.csv \
      --edge-url http://raspberrypi.local:8101 --n 300
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
from pathlib import Path

import httpx
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--offline", required=True)
    ap.add_argument("--edge-url", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    off = pd.read_csv(args.offline)
    off["key"] = off["true_label"] + "/" + off["image_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    off = off.set_index("key")
    root = Path(args.image_dir)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    imgs = imgs[:: max(1, len(imgs) // args.n)][: args.n]

    agree, dm, edge_ms = 0, [], []
    with httpx.Client(base_url=args.edge_url, timeout=60) as c:
        for p in imgs:
            key = f"{p.parent.name}/{p.name}"
            r = c.post("/predict", json={"image_b64": base64.b64encode(p.read_bytes()).decode("ascii"),
                                         "force_route": "edge"}).json()
            ep = r["edge_prediction"]
            agree += ep["label"] == off.loc[key, "edge_pred"]
            dm.append(abs(ep["margin"] - float(off.loc[key, "edge_margin"])))
            edge_ms.append(float(r.get("edge_ms", 0.0)))
    rep = {"n": len(imgs), "label_agreement": agree / len(imgs),
           "margin_abs_diff_mean": statistics.mean(dm), "margin_abs_diff_max": max(dm),
           "edge_ms_median": statistics.median(edge_ms)}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

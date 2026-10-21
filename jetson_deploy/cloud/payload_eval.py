"""Cloud predictions from the original photograph and from candidate payloads.

Runs the reference cloud pipeline (PyTorch backbones + NumPy head, the one
validated by export_onnx.py) on a list of images: the original file and each
payload policy (shorter side, JPEG quality). Writes one row per image with the
byte sizes and every prediction, and a summary per policy.

Used for (a) choosing the payload policy on the VALIDATION split and (b)
separating the payload's effect from the Nano's on the escalated images.

  python jetson_deploy/cloud/payload_eval.py --list <csv with image_path,true_label> \
      --trt-dir outputs/trt_export --configs 448:90,336:95,336:90,224:90 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import trt_common as tc  # noqa: E402
from cloud.export_onnx import build  # noqa: E402
from shared.image_io import encode_jpeg, escalation_payload  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="CSV with image_path and true_label columns")
    ap.add_argument("--image-root", default="", help="prefix for relative image paths")
    ap.add_argument("--trt-dir", required=True)
    ap.add_argument("--configs", default="448:90,336:95,336:90,224:90")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.list, encoding="utf-8")))
    paths = [Path(args.image_root) / r["image_path"] if args.image_root else Path(r["image_path"]) for r in rows]
    truth = [r["true_label"] for r in rows]
    cfgs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
    tags = ["original", "full_q90"] + [f"{s}px_q{q}" for s, q in cfgs]
    d = Path(args.trt_dir)
    head = tc.QualityGateNP(str(d / "qualitygate_head.npz"))
    classes = tc.load_class_names(str(d / "class_names.txt"))
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # payload bytes, encoded once
    kb = {t: [] for t in tags}
    blobs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        kb["original"].append(p.stat().st_size / 1024)
        b = {"original": None}
        raw, k = encode_jpeg(im, quality=90)
        b["full_q90"] = raw
        kb["full_q90"].append(k)
        for (s, q), t in zip(cfgs, tags[2:]):
            raw, k = encode_jpeg(escalation_payload(im, s), quality=q)
            b[t] = raw
            kb[t].append(k)
        blobs.append(b)

    feats = {t: [] for t in tags}
    for spec in tc.BACKBONES:
        m = build(spec["name"], gpu)
        per = {t: [] for t in tags}
        with torch.no_grad():
            for p, b in zip(paths, blobs):
                for t in tags:
                    im = Image.open(p if b[t] is None else io.BytesIO(b[t])).convert("RGB")
                    x = torch.from_numpy(tc.preprocess(im, spec["size"], spec["mean"], spec["std"])).to(gpu)
                    per[t].append(m(x).cpu().numpy())
        for t in tags:
            feats[t].append(np.concatenate(per[t]))
        del m
        torch.cuda.empty_cache()
        print("done", spec["name"], flush=True)

    pred = {t: [classes[i] for i in tc.softmax(head(tc.fuse(v))).argmax(-1)] for t, v in feats.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "true_label"] + [f"pred_{t}" for t in tags] + [f"kb_{t}" for t in tags])
        for i, r in enumerate(rows):
            w.writerow([r["image_path"], truth[i]] + [pred[t][i] for t in tags] + [f"{kb[t][i]:.3f}" for t in tags])
    summ = {"n": len(rows), "list": args.list, "policies": {}}
    for t in tags:
        summ["policies"][t] = {
            "kb_mean": statistics.mean(kb[t]), "kb_median": statistics.median(kb[t]),
            "acc": float(np.mean([a == y for a, y in zip(pred[t], truth)])),
            "agreement_with_original": float(np.mean([a == b for a, b in zip(pred[t], pred["original"])]))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

"""Does the cloud give the same diagnosis from the escalation payload
(shorter side 336 px, JPEG q90) as from the original photograph?

Runs the reference cloud pipeline (PyTorch backbones + numpy head, as
validated by export_onnx.py) on N test images twice -- original file vs.
payload -- and reports prediction agreement, accuracy and byte sizes.

  python jetson_deploy/cloud/check_payload_fidelity.py --image-dir <plantwild_test> \
      --trt-dir outputs/trt_export --n 300
"""

from __future__ import annotations

import argparse
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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--trt-dir", required=True, help="has qualitygate_head.npz, class_names.txt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--short-side", type=int, default=336)
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    root, d = Path(args.image_dir), Path(args.trt_dir)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    imgs = imgs[:: max(1, len(imgs) // args.n)][: args.n]
    head = tc.QualityGateNP(str(d / "qualitygate_head.npz"))
    classes = tc.load_class_names(str(d / "class_names.txt"))
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    originals, payloads, kb_orig_file, kb_full_q, kb_payload = [], [], [], [], []
    for p in imgs:
        im = Image.open(p).convert("RGB")
        originals.append(im)
        _, kb_full = encode_jpeg(im, quality=args.quality)
        raw, kb = encode_jpeg(escalation_payload(im, args.short_side), quality=args.quality)
        payloads.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        kb_orig_file.append(p.stat().st_size / 1024)
        kb_full_q.append(kb_full)
        kb_payload.append(kb)

    feats = {"orig": [], "payload": []}
    for spec in tc.BACKBONES:
        m = build(spec["name"], gpu)
        with torch.no_grad():
            for key, ims in (("orig", originals), ("payload", payloads)):
                f = [m(torch.from_numpy(tc.preprocess(im, spec["size"], spec["mean"], spec["std"])).to(gpu)).cpu().numpy()
                     for im in ims]
                feats[key].append(np.concatenate(f))
        del m
        torch.cuda.empty_cache()

    pred = {k: [classes[i] for i in tc.softmax(head(tc.fuse(v))).argmax(-1)] for k, v in feats.items()}
    truth = [p.parent.name for p in imgs]
    rep = {
        "n": len(imgs), "short_side": args.short_side, "jpeg_quality": args.quality,
        "agreement": float(np.mean([a == b for a, b in zip(pred["orig"], pred["payload"])])),
        "acc_original": float(np.mean([a == t for a, t in zip(pred["orig"], truth)])),
        "acc_payload": float(np.mean([a == t for a, t in zip(pred["payload"], truth)])),
        "kb_original_file_mean": statistics.mean(kb_orig_file),
        "kb_fullres_q_mean": statistics.mean(kb_full_q),
        "kb_payload_mean": statistics.mean(kb_payload),
        "kb_payload_median": statistics.median(kb_payload),
    }
    print(json.dumps(rep, indent=2))
    (d / "payload_fidelity.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

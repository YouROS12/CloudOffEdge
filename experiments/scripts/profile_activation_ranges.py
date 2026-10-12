"""Largest activation magnitudes per transformer block, for each cloud backbone.

Explains the per-backbone precision of the Nano port: fp16 overflows above
65,504. Hooks every transformer block output (the residual stream) and every
nn.Linear output of the fp32 reference models, over test images, and records
max |x| per block. Runs on the workstation GPU.

  python experiments/scripts/profile_activation_ranges.py --n 64
  -> results/activation_ranges.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
sys.path[:0] = [str(PAPER2 / "rasp_deploy"), str(PAPER2 / "jetson_deploy")]
from cloud import trt_common as tc  # noqa: E402
from cloud.export_onnx import Chunk, build  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FP16_MAX = 65504.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default=str(PAPER2 / "data" / "plantwild_test"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=str(PAPER2 / "results" / "activation_ranges.json"))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    paths = paths[:: max(1, len(paths) // args.n)][: args.n]

    report = {"fp16_max": FP16_MAX, "n_images": len(paths), "device": str(dev), "backbones": {}}
    for spec in tc.BACKBONES:
        name = spec["name"]
        feat = build(name, dev).float()
        blocks = Chunk(name, feat.fn, 0, 0, False, False).blocks()
        resid = [0.0] * len(blocks)
        lin = [0.0] * len(blocks)
        hooks = []
        for i, blk in enumerate(blocks):
            def on_block(_m, _inp, out, i=i):
                o = out[0] if isinstance(out, tuple) else out
                resid[i] = max(resid[i], float(o.detach().abs().max()))
            hooks.append(blk.register_forward_hook(on_block))
            for sub in blk.modules():
                if isinstance(sub, torch.nn.Linear):
                    def on_lin(_m, _inp, out, i=i):
                        lin[i] = max(lin[i], float(out.detach().abs().max()))
                    hooks.append(sub.register_forward_hook(on_lin))
        with torch.no_grad():
            for p in paths:
                x = torch.from_numpy(tc.preprocess(Image.open(p).convert("RGB"), spec["size"],
                                                   spec["mean"], spec["std"])).to(dev)
                feat(x)
        for h in hooks:
            h.remove()
        report["backbones"][name] = {"residual_max_per_block": resid, "linear_max_per_block": lin,
                                     "residual_max": max(resid), "linear_max": max(lin),
                                     "blocks_over_fp16": sum(r > FP16_MAX for r in resid)}
        print(f"{name}: residual max {max(resid):,.0f}, linear max {max(lin):,.0f}, "
              f"{report['backbones'][name]['blocks_over_fp16']} blocks over fp16")
        del feat
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

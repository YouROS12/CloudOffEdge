"""Export each cloud backbone as K consecutive ONNX chunks for the Nano.

TensorRT 8.2's ONNX parser needs ~4x the graph size in RAM (2.75 GB peak to
parse a 607 MB fp16 ViT-L), so a whole backbone cannot be built on a 4 GB
Jetson Nano. Chunk i holds transformer blocks [s_i, e_i) -- plus the stem
for i = 0 and the pooling/projection tail for i = K-1 -- and the chain of
chunks computes exactly the full forward.

  python jetson_deploy/cloud/export_chunks.py --out outputs/trt_export --chunks 4 \
      --image-dir <plantwild_test>
  -> <name>.c<i>.onnx (fp32, simplified) ; then cloud/onnx_to_fp16.py

Checks: torch chain == torch full model; onnxruntime chain == torch full.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import trt_common as tc  # noqa: E402
from cloud.export_onnx import (_ORIG_SDPA, build, cos, linear_as_conv, make_chunks,  # noqa: E402
                               manual_sdpa, simplify)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--no-linear-as-conv", action="store_true",
                    help="keep nn.Linear as MatMul (TensorRT 8.2 then puts the weights in Myelin)")
    ap.add_argument("--only", default="", help="export just this backbone")
    ap.add_argument("--only-chunk", type=int, default=-1, help="export just this chunk index")
    args = ap.parse_args()

    import onnxruntime as ort

    out = Path(args.out)
    imgs = sorted(p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)[::900][:3]
    report = {}
    for spec in tc.BACKBONES:
        name, size = spec["name"], spec["size"]
        if args.only and name != args.only:
            continue
        full = build(name, torch.device("cpu"))
        xs = [torch.from_numpy(tc.preprocess(Image.open(p).convert("RGB"), size, spec["mean"], spec["std"]))
              for p in imgs]
        with torch.no_grad():
            ref = torch.cat([full(x) for x in xs]).numpy()   # reference BEFORE any rewrite
        if not args.no_linear_as_conv:
            linear_as_conv(full)
        chunks = make_chunks(name, full, args.chunks)

        with torch.no_grad():
            h = xs
            for c in chunks:
                h = [c(x) for x in h]
            chain = torch.cat(h).numpy()
        c_torch = float(cos(chain, ref).min())

        # export each chunk with a representative input (output of the previous chunk)
        paths, x_in = [], xs[0]
        F.scaled_dot_product_attention = manual_sdpa
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for i, c in enumerate(chunks):
                p = out / f"{name}.c{i}.onnx"
                if 0 <= args.only_chunk != i:
                    with torch.no_grad():
                        x_in = c(x_in)
                    continue
                t0 = time.time()
                torch.onnx.export(c, x_in, str(p), input_names=["input"], output_names=["out"],
                                  opset_version=args.opset, do_constant_folding=True, dynamo=False)
                bad = simplify(p)
                if bad:
                    raise SystemExit(f"{p.name}: unsupported ops remain: {bad}")
                with torch.no_grad():
                    x_in = c(x_in)
                print(f"  {p.name}: blocks [{c.s},{c.e}) -> {tuple(x_in.shape)}, "
                      f"{p.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s")
                paths.append(p)
        finally:
            F.scaled_dot_product_attention = _ORIG_SDPA
            torch.backends.mha.set_fastpath_enabled(True)

        print(f"{name}: torch chain (linear-as-conv={not args.no_linear_as_conv}) vs full cos {c_torch:.6f}")
        if args.only_chunk >= 0:
            continue
        sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in paths]
        outs = []
        for x in xs:
            h = x.numpy()
            for s in sessions:
                h = s.run(None, {"input": h})[0]
            outs.append(h)
        c_ort = float(cos(np.concatenate(outs), ref).min())
        print(f"{name}: {args.chunks} chunks | torch chain vs full cos {c_torch:.6f} | "
              f"onnxruntime chain vs full cos {c_ort:.6f}")
        report[name] = {"chunks": args.chunks, "blocks": [[c.s, c.e] for c in chunks],
                        "torch_chain_cos": c_torch, "onnx_chain_cos": c_ort}
        del full, chunks, sessions
    (out / "chunks_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

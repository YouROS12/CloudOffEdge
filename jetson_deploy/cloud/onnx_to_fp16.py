"""Convert the exported fp32 ONNX backbones to fp16 weights for the Nano.

Building a TensorRT engine from a 1.2 GB fp32 ViT-L graph fails on the
4 GB Jetson Nano (std::bad_alloc in the builder: GPU memory is carved out
of system RAM and cannot be swapped). The engine runs in fp16 anyway, so
the graph is converted to fp16 weights beforehand, halving the builder's
footprint. LayerNorm arithmetic (Pow / ReduceMean / Sqrt / Div) is kept in
fp32: ViT activations contain large outliers whose squares overflow fp16.

Chunked graphs (<name>.c<i>.onnx from cloud/export_chunks.py) are converted
when present; otherwise the whole graph <name>.onnx.

  python jetson_deploy/cloud/onnx_to_fp16.py --dir outputs/trt_export
      -> <stem>.fp16.onnx next to each source graph
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import onnx
from onnxconverter_common import float16

NAMES = ["dinov2_vitl14", "dinov3_vitl16", "clip_vitl14"]
KEEP_FP32 = ["Pow", "ReduceMean", "Sqrt", "Div"]


def sources(d: Path, name: str) -> list[Path]:
    chunks = [p for p in d.glob(f"{name}.c*.onnx") if not p.name.endswith(".fp16.onnx")]
    if chunks:
        return sorted(chunks, key=lambda p: int(p.name.rsplit(".c", 1)[1].split(".")[0]))
    return [d / f"{name}.onnx"]


def convert(src: Path) -> None:
    dst = src.with_name(src.name[: -len(".onnx")] + ".fp16.onnx")
    m16 = float16.convert_float_to_float16(
        onnx.load(str(src)), keep_io_types=True,
        op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + KEEP_FP32)
    onnx.save(m16, str(dst))
    casts = collections.Counter(n.op_type for n in m16.graph.node)["Cast"]
    print(f"{src.name}: {src.stat().st_size / 1e6:.0f} MB -> {dst.name} {dst.stat().st_size / 1e6:.0f} MB "
          f"({casts} Cast nodes around fp32 LayerNorm math)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    for name in NAMES:
        for src in sources(d, name):
            convert(src)


if __name__ == "__main__":
    main()

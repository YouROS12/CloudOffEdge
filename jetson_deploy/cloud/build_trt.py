"""Build fp16 TensorRT engines from the exported ONNX backbones ON the
Jetson Nano, then gate them against the desktop reference features.

Each engine is built in its own subprocess so the builder's memory is
fully released before the next one (4 GB shared CPU/GPU memory).

  python build_trt.py --trt-dir /opt/plant-deploy/trt            # build + verify
  python build_trt.py --trt-dir ... --verify-only
  python build_trt.py --trt-dir ... --only dinov2_vitl14 --precision fp16-ln32

--precision:
  fp16       everything fp16 (fastest)
  fp16-ln32  fp16, but LayerNorm math (reduce / pow / sqrt / div) kept in
             fp32 -- the usual fix when large ViT activations overflow fp16
  fp32       reference / fallback
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cloud import trt_common as tc  # noqa: E402


def build_one(onnx_path, engine_path, precision, workspace_mb):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    # parse_from_file, not parse(fh.read()): reading the 1.2 GB graph into a
    # Python bytes object first holds the weights twice and gets the build
    # OOM-killed on the 4 GB Nano.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  parse error:", parser.get_error(i))
        raise SystemExit(1)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb << 20
    if os.environ.get("TRT_TACTICS") == "cublas":
        # Memory-saving option: time only TensorRT's own kernels + cuBLAS
        # (skips loading cuDNN / cuBLASLt). Not needed for chunked builds,
        # and it produced 4x larger engines (606 MB per 6-block chunk).
        config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    if precision != "fp32":
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "fp16-ln32":
        # With the FP16 flag TensorRT may run LayerNorm in fp16 even though the
        # graph casts it to fp32; squaring ViT activation outliers (>256) then
        # overflows (inf -> NaN; observed for DINOv3). Pin every reduction and
        # every elementwise/unary layer inside a LayerNorm scope to fp32.
        flag = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None) or trt.BuilderFlag.STRICT_TYPES
        config.set_flag(flag)
        ln_scopes = ("norm", "/ln_", "ln_pre", "ln_post")
        n = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            in_ln = any(s in layer.name for s in ln_scopes)
            if layer.type == trt.LayerType.REDUCE or (
                    in_ln and layer.type in (trt.LayerType.ELEMENTWISE, trt.LayerType.UNARY)):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                n += 1
        print("  pinned %d LayerNorm layers to fp32" % n, flush=True)
    if precision == "all32":
        # Full fp32 arithmetic from the fp16-weight graph. The graph carries
        # fp16-typed tensors (the converter's casts), which TensorRT rejects
        # unless FP16 is enabled -- so FP16 stays on and every layer is pinned
        # to fp32. Needed for DINOv3: its residual stream reaches ~1.6e5,
        # beyond the fp16 range (65504).
        flag = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None) or trt.BuilderFlag.STRICT_TYPES
        config.set_flag(flag)
        float_types = (trt.float32, trt.float16)
        n = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            outs = [layer.get_output(j) for j in range(layer.num_outputs)]
            if outs and all(o.dtype in float_types for o in outs):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                n += 1
        print("  all32: %d layers pinned to fp32" % n, flush=True)
    if precision == "conv16":
        # Only the convolutions (the linear layers, ~95% of the FLOPs) run in
        # fp16; attention products, softmax, LayerNorm, GELU and the residual
        # stream stay fp32. Needed because fp16 elsewhere degraded CLIP
        # (min cosine 0.947) even though its activations are small.
        flag = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None) or trt.BuilderFlag.STRICT_TYPES
        config.set_flag(flag)
        float_types = (trt.float32, trt.float16)
        n = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type == trt.LayerType.CONVOLUTION:
                continue
            outs = [layer.get_output(j) for j in range(layer.num_outputs)]
            if outs and all(o.dtype in float_types for o in outs):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                n += 1
        print("  conv16: %d non-convolution layers pinned to fp32" % n, flush=True)
    # Shared timing cache: all chunks of a backbone have identical layer
    # shapes, so kernel timings measured for the first chunk (~10 min on the
    # Nano) are reused by the others.
    cache_path = os.path.join(os.path.dirname(engine_path), "timing.cache")
    blob = open(cache_path, "rb").read() if os.path.exists(cache_path) else b""
    config.set_timing_cache(config.create_timing_cache(blob), False)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("  build failed")
    with open(cache_path, "wb") as fh:
        fh.write(config.get_timing_cache().serialize())
    with open(engine_path, "wb") as fh:
        fh.write(serialized)
    print("  built %s (%.0f MB) in %.0f s" % (os.path.basename(engine_path),
                                            os.path.getsize(engine_path) / 1e6, time.time() - t0))


def build_units(trt_dir, name):
    """Engines to build for one backbone: its chunks (<name>.c<i>) if the
    chunked graphs exist (cloud/export_chunks.py), otherwise the whole graph."""
    import glob
    chunks = sorted(glob.glob(os.path.join(trt_dir, name + ".c*.fp16.onnx")),
                    key=lambda p: int(p.rsplit(".c", 1)[1].split(".")[0]))
    if chunks:
        return [os.path.basename(p)[: -len(".fp16.onnx")] for p in chunks]
    return [name]


def verify(trt_dir, n_timing):
    from PIL import Image
    from cloud.trt_runtime import run_backbone_batched
    refs = json.load(open(os.path.join(trt_dir, "ref_images.json")))
    img_root = os.path.join(trt_dir, "ref_images")
    head = tc.QualityGateNP(os.path.join(trt_dir, "qualitygate_head.npz"))
    classes = tc.load_class_names(os.path.join(trt_dir, "class_names.txt"))
    feats, report = [], {}
    for spec in tc.BACKBONES:
        name = spec["name"]
        xs = [tc.preprocess(Image.open(os.path.join(img_root, r["image"])).convert("RGB"),
                            spec["size"], spec["mean"], spec["std"]) for r in refs]
        # chunk-major, exactly as the cloud server's batch flush runs it
        outs, load_ms, comp = run_backbone_batched(trt_dir, name, xs)
        f = np.concatenate(outs)
        ref = np.load(os.path.join(trt_dir, "ref_" + name + ".npy"))
        np.save(os.path.join(trt_dir, "trt_" + name + ".npy"), f)  # for offline analysis
        cos = (tc.l2_normalize(f) * tc.l2_normalize(ref)).sum(-1)
        ms = float(np.median(comp[1:] if len(comp) > 1 else comp))  # first image includes warm-up
        print("  %-14s cosine vs desktop: min %.4f mean %.4f | %.0f ms/img | load %.1f s"
              % (name, np.nanmin(cos), np.nanmean(cos), ms, load_ms / 1000), flush=True)
        report[name] = {"cos_min": float(np.nanmin(cos)), "cos_mean": float(np.nanmean(cos)),
                        "nan": int(np.isnan(cos).sum()), "ms": ms, "load_ms": load_ms}
        feats.append(f)
    preds = tc.softmax(head(tc.fuse(feats)))
    agree = sum(classes[int(p.argmax())] == r["pred"] for p, r in zip(preds, refs))
    print("  top-1 agreement with desktop pipeline: %d/%d" % (agree, len(refs)))
    report["top1_agree"] = agree / float(len(refs))
    json.dump(report, open(os.path.join(trt_dir, "verify_report.json"), "w"), indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trt-dir", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp16-ln32", "conv16", "all32", "fp32"])
    ap.add_argument("--workspace-mb", type=int, default=256)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--n-timing", type=int, default=10)
    ap.add_argument("--rebuild", action="store_true", help="rebuild engines that already exist")
    args = ap.parse_args()

    if args.only:
        # Prefer the fp16-weight graph (cloud/onnx_to_fp16.py): the fp32 one
        # exhausts the Nano's memory during the build.
        fp16 = os.path.join(args.trt_dir, args.only + ".fp16.onnx")
        src = fp16 if os.path.exists(fp16) else os.path.join(args.trt_dir, args.only + ".onnx")
        print("  source graph:", os.path.basename(src), flush=True)
        build_one(src, os.path.join(args.trt_dir, args.only + ".engine"),
                  args.precision, args.workspace_mb)
        return
    if not args.verify_only:
        for spec in tc.BACKBONES:
            for unit in build_units(args.trt_dir, spec["name"]):
                if not args.rebuild and os.path.exists(os.path.join(args.trt_dir, unit + ".engine")):
                    print("[skip] %s (engine exists)" % unit, flush=True)
                    continue
                print("[build] %s (%s)" % (unit, args.precision), flush=True)
                subprocess.check_call([sys.executable, os.path.abspath(__file__), "--trt-dir", args.trt_dir,
                                       "--only", unit, "--precision", args.precision,
                                       "--workspace-mb", str(args.workspace_mb)])
    print("[verify]")
    verify(args.trt_dir, args.n_timing)


if __name__ == "__main__":
    main()

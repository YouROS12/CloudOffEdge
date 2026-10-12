"""
Compute params / FLOPs / on-device latency for the three edge backbones used in
Paper 2 (v3-Small, v3-Large, EfficientNet-B0). Mirrors the build_model() factory
from train_edge_model.py at 224x224, 89 PlantWild classes.

Latency is measured on whatever device torch picks (cuda if available, else cpu)
with a configurable batch size; we report both single-image and batch-amortized
ms. Outputs CSV + JSON to results/option_A_3arch/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_edge_model import build_model  # type: ignore

ARCHS = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"]
LABELS = {
    "mobilenet_v3_small": "v3-Small",
    "mobilenet_v3_large": "v3-Large",
    "efficientnet_b0":    "EfficientNet-B0",
}


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def count_flops(m: torch.nn.Module, x: torch.Tensor) -> int:
    """Use fvcore for FLOPs. Returns total MACs as reported (these are MACs)."""
    from fvcore.nn import FlopCountAnalysis
    fa = FlopCountAnalysis(m, x)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return int(fa.total())


@torch.no_grad()
def measure_latency(m: torch.nn.Module, x: torch.Tensor, *,
                    warmup: int = 20, iters: int = 100) -> dict:
    device = x.device
    is_cuda = device.type == "cuda"

    # warmup
    for _ in range(warmup):
        m(x)
        if is_cuda:
            torch.cuda.synchronize()

    # timed
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        m(x)
        if is_cuda:
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    batch = x.shape[0]
    total_imgs = iters * batch
    return {
        "batch": batch,
        "iters": iters,
        "total_seconds": elapsed,
        "ms_per_batch": (elapsed / iters) * 1000,
        "ms_per_image": (elapsed / total_imgs) * 1000,
        "imgs_per_sec": total_imgs / elapsed,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/option_A_3arch")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_classes = 89
    img_size = 224
    batch_sizes = [1, 32]

    rows = []
    print(f"\n{'='*78}")
    print(f"Edge architecture benchmark (device={device}, img={img_size}x{img_size}, "
          f"n_classes={n_classes})")
    print('='*78)

    for arch in ARCHS:
        m = build_model(arch, n_classes).to(device).eval()
        params = count_params(m)

        # FLOPs at batch=1 (fvcore wants a real tensor)
        x1 = torch.randn(1, 3, img_size, img_size, device=device)
        try:
            flops = count_flops(m, x1)
            mflops = flops / 1e6
            gflops = flops / 1e9
        except Exception as e:
            print(f"[warn] FLOPs failed for {arch}: {e}")
            mflops = float("nan")
            gflops = float("nan")

        row = {
            "arch": LABELS[arch],
            "torch_arch_id": arch,
            "params_M": params / 1e6,
            "flops_M_macs_b1": mflops,
            "flops_G_macs_b1": gflops,
        }

        for bs in batch_sizes:
            x = torch.randn(bs, 3, img_size, img_size, device=device)
            lat = measure_latency(m, x)
            row[f"ms_per_image_b{bs}"] = lat["ms_per_image"]
            row[f"ms_per_batch_b{bs}"] = lat["ms_per_batch"]
            row[f"imgs_per_sec_b{bs}"] = lat["imgs_per_sec"]

        rows.append(row)
        print(f"\n[{LABELS[arch]:>15}] params={params/1e6:6.2f} M  "
              f"FLOPs={gflops:6.3f} G-macs  "
              f"b1={row['ms_per_image_b1']:6.2f} ms/img  "
              f"b32={row['ms_per_image_b32']:6.2f} ms/img  "
              f"({row['imgs_per_sec_b32']:7.0f} img/s)")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = out_dir / "arch_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[write] {csv_path}")

    # JSON
    json_path = out_dir / "arch_benchmark.json"
    json_path.write_text(json.dumps({
        "device": str(device),
        "img_size": img_size,
        "n_classes": n_classes,
        "rows": rows,
    }, indent=2))
    print(f"[write] {json_path}")


if __name__ == "__main__":
    main()

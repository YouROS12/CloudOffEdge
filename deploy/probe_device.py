#!/usr/bin/env python3
"""
CloudOffEdge -- device prober for the paper's hardware section.

Run this ON the Raspberry Pi (edge) and ON the Jetson Nano (cloud). It collects
real hardware/OS/torch specs and, when torch is installed, measures:

  * sustained fp32 GFLOPS (matmul micro-benchmark)  -> device_specs.achieved_gflops_fp32
  * edge only: real MobileNetV3-Large forward latency -> measured_overrides.json

It writes probe_<hostname>.json and prints a paste-ready snippet. Copy that JSON
back to your dev machine so hardware_projection/ can flip ESTIMATE -> MEASURED.

Python-only, no extra deps. Degrades gracefully if torch is missing (specs only).

Usage:
    python3 deploy/probe_device.py                 # autodetect role, img-size 224
    python3 deploy/probe_device.py --role edge
    python3 deploy/probe_device.py --img-size 224 --runs 50
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# System spec collection (stdlib only)
# ---------------------------------------------------------------------------
def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").strip()
    except Exception:
        return ""


def detect_role() -> str:
    if Path("/etc/nv_tegra_release").exists():
        return "cloud"
    model = _read("/proc/device-tree/model") or _read("/sys/firmware/devicetree/base/model")
    if "raspberry pi" in model.lower():
        return "edge"
    return "unknown"


def device_model() -> str:
    m = _read("/proc/device-tree/model") or _read("/sys/firmware/devicetree/base/model")
    if m:
        return m.replace("\x00", "").strip()
    tegra = _read("/etc/nv_tegra_release")
    if tegra:
        return f"NVIDIA Jetson ({tegra.splitlines()[0]})"
    return platform.platform()


def device_key(role: str, model: str) -> str:
    ml = model.lower()
    if role == "edge":
        if "raspberry pi 5" in ml:
            return "pi5"
        if "raspberry pi 4" in ml:
            return "pi4"
        return "pi_unknown"
    if role == "cloud":
        return "nano"
    return "unknown"


def cpu_info() -> dict:
    name, hardware = "", ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name") and not name:
            name = line.split(":", 1)[-1].strip()
        if line.lower().startswith(("hardware", "model")) and not hardware:
            hardware = line.split(":", 1)[-1].strip()
    max_khz = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    return {
        "model_name": name or hardware or platform.processor() or "unknown",
        "cores": os.cpu_count(),
        "max_freq_ghz": round(int(max_khz) / 1e6, 2) if max_khz.isdigit() else None,
        "arch": platform.machine(),
    }


def mem_total_gb() -> float | None:
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal"):
            kb = int("".join(c for c in line if c.isdigit()))
            return round(kb / 1024 / 1024, 2)
    return None


def os_info() -> dict:
    pretty = ""
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            pretty = line.split("=", 1)[-1].strip().strip('"')
    return {"os": pretty or platform.platform(), "kernel": platform.release()}


# ---------------------------------------------------------------------------
# Torch-based benchmarks (optional)
# ---------------------------------------------------------------------------
def torch_info():
    try:
        import torch  # noqa
        info = {"available": True, "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
        try:
            import torchvision
            info["torchvision"] = torchvision.__version__
        except Exception:
            info["torchvision"] = None
        return info
    except Exception as e:
        return {"available": False, "error": str(e)}


def bench_matmul_gflops(runs: int = 20) -> dict | None:
    """Sustained fp32 GFLOPS via NxN matmul (CPU, and GPU if present)."""
    try:
        import torch
    except Exception:
        return None
    out = {}
    for label, dev in [("cpu", "cpu"), ("cuda", "cuda")]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        n = 1024 if dev == "cpu" else 2048
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        for _ in range(3):  # warmup
            (a @ b)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            c = a @ b
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs
        flops = 2.0 * n ** 3
        out[label] = round(flops / dt / 1e9, 1)  # GFLOPS
    return out or None


def bench_edge_latency(img_size: int, runs: int) -> dict | None:
    """Real MobileNetV3-Large fp32 forward latency (the edge model's backbone)."""
    try:
        import torch
        from torchvision.models import mobilenet_v3_large
    except Exception:
        return None
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(os.cpu_count() or 4)
    except Exception:
        pass
    model = mobilenet_v3_large(weights=None).eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):  # warmup
        model(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    ms = (time.perf_counter() - t0) / runs * 1000.0
    return {"model": "mobilenet_v3_large", "img_size": img_size,
            "runs": runs, "ms_per_image_fp32": round(ms, 2)}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Probe a CloudOffEdge device.")
    ap.add_argument("--role", choices=["edge", "cloud"], default=None,
                    help="override autodetected role")
    ap.add_argument("--img-size", type=int, default=224,
                    help="input size for the edge latency benchmark")
    ap.add_argument("--runs", type=int, default=50,
                    help="timed iterations per benchmark")
    ap.add_argument("--no-bench", action="store_true",
                    help="collect specs only, skip torch benchmarks")
    args = ap.parse_args()

    role = args.role or detect_role()
    model = device_model()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "role": role,
        "device_key": device_key(role, model),
        "device_model": model,
        "cpu": cpu_info(),
        "ram_total_gb": mem_total_gb(),
        **os_info(),
        "python": platform.python_version(),
        "torch": torch_info(),
    }

    if not args.no_bench and report["torch"].get("available"):
        gflops = bench_matmul_gflops(runs=max(10, args.runs // 2))
        if gflops:
            report["measured_gflops_fp32"] = gflops
        if role == "edge":
            lat = bench_edge_latency(args.img_size, args.runs)
            if lat:
                report["edge_latency"] = lat

    out_path = Path.cwd() / f"probe_{report['hostname']}.json"
    out_path.write_text(json.dumps(report, indent=2))

    # ---- human summary ----
    print("=" * 62)
    print(f" CloudOffEdge probe -- {report['hostname']}  (role={role})")
    print("=" * 62)
    print(f" device : {model}")
    print(f" cpu    : {report['cpu']['model_name']} x{report['cpu']['cores']} "
          f"@ {report['cpu']['max_freq_ghz']} GHz ({report['cpu']['arch']})")
    print(f" ram    : {report['ram_total_gb']} GB")
    print(f" os     : {report['os']}")
    print(f" python : {report['python']}")
    tv = report["torch"]
    if tv.get("available"):
        print(f" torch  : {tv['version']} (cuda={tv['cuda_available']})")
    else:
        print(f" torch  : NOT INSTALLED ({tv.get('error', '')[:50]}) -- specs only")
    if "measured_gflops_fp32" in report:
        print(f" gflops : {report['measured_gflops_fp32']} (fp32, matmul)")
    if "edge_latency" in report:
        print(f" edge   : {report['edge_latency']['ms_per_image_fp32']} ms/image "
              f"(MobileNetV3-L fp32 @ {args.img_size})")
    print("-" * 62)
    print(f" wrote  : {out_path}")

    # ---- paste-ready snippet for hardware_projection ----
    key = report["device_key"]
    src = f"real {model}, torch {tv.get('version','?')} fp32, mean of {args.runs} runs, {report['timestamp_utc']}"
    if role == "edge" and "edge_latency" in report:
        snippet = {"edge": {key: {"v3-Large": {
            "ms": report["edge_latency"]["ms_per_image_fp32"], "source": src}}}}
        print("\n Paste into hardware_projection/measured_overrides.json (edge):")
        print(json.dumps(snippet, indent=2))
    elif role == "cloud":
        print("\n Cloud note: real resolve latency is best captured from the deploy"
              "\n server's own stats during the batch run (3 ViT-L backbones are too"
              "\n heavy to blind-load here). Specs + GFLOPS above are recorded.")
    print("=" * 62)


if __name__ == "__main__":
    main()

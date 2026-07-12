"""Batch client: walk a folder of images, POST each to the edge server,
log per-image results to a CSV.

Usage:
  python batch_predict.py \
    --image-dir /path/to/PlantWild/test/images \
    --edge-url http://<pi-tailscale-name>:8000 \
    --out-csv results/run_$(date +%F).csv

CSV columns:
  image_path, edge_label, edge_confidence, edge_margin, edge_entropy,
  final_label, source, routed, bandwidth_kb, latency_ms

The edge server already does the routing decision — this client just
posts and logs. Pass --label-from-dirname to read true labels from the
parent folder name (PlantWild test layout: <root>/<class_name>/img.jpg).
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.image_io import encode_b64  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--edge-url", default="http://localhost:8000")
    ap.add_argument("--out-csv", default="results/batch_predictions.csv")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N images (0 = all)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--label-from-dirname", action="store_true",
                    help="Use parent directory name as the true label.")
    args = ap.parse_args()

    root = Path(args.image_dir)
    if not root.exists():
        raise SystemExit(f"image dir not found: {root}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_path", "true_label",
        "edge_label", "edge_confidence", "edge_margin", "edge_entropy",
        "final_label", "source", "routed", "bandwidth_kb", "latency_ms",
    ]

    n_total = 0
    n_routed = 0
    bw_sum = 0.0
    lat_sum = 0.0
    n_correct = 0
    n_with_label = 0

    t0 = time.time()
    client = httpx.Client(base_url=args.edge_url, timeout=args.timeout)
    try:
        # Probe health first so we fail fast on a misconfigured CLOUD_URL.
        r = client.get("/health")
        r.raise_for_status()
        print(f"[health] {r.json()}")
    except Exception as e:
        raise SystemExit(f"edge server unreachable at {args.edge_url}: {e}")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for img_path in iter_images(root):
            if args.limit and n_total >= args.limit:
                break
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[skip] {img_path}: {e}")
                continue
            b64, _ = encode_b64(img, quality=args.jpeg_quality)
            true_label = img_path.parent.name if args.label_from_dirname else ""
            try:
                r = client.post("/predict", json={
                    "image_b64": b64,
                    "request_id": img_path.stem,
                })
                r.raise_for_status()
                resp = r.json()
            except Exception as e:
                print(f"[err] {img_path.name}: {e}")
                continue

            ep = resp["edge_prediction"]
            row = {
                "image_path": str(img_path),
                "true_label": true_label,
                "edge_label": ep["label"],
                "edge_confidence": f"{ep['confidence']:.6f}",
                "edge_margin": f"{ep['margin']:.6f}",
                "edge_entropy": f"{ep['entropy']:.6f}",
                "final_label": resp["label"],
                "source": resp["source"],
                "routed": int(resp["routed"]),
                "bandwidth_kb": f"{resp['bandwidth_kb']:.3f}",
                "latency_ms": f"{resp['latency_ms']:.1f}",
            }
            w.writerow(row)
            f.flush()

            n_total += 1
            if resp["routed"]:
                n_routed += 1
            bw_sum += resp["bandwidth_kb"]
            lat_sum += resp["latency_ms"]
            if true_label:
                n_with_label += 1
                if resp["label"] == true_label:
                    n_correct += 1

            if n_total % 25 == 0:
                avg_bw = bw_sum / n_total
                avg_lat = lat_sum / n_total
                rate = n_routed / n_total * 100
                acc = (n_correct / n_with_label * 100) if n_with_label else float("nan")
                print(f"[{n_total:>5}] route={rate:5.1f}%  "
                      f"avg_bw={avg_bw:6.2f} KB  avg_lat={avg_lat:6.0f} ms  "
                      f"acc={acc:5.2f}%")

    dt = time.time() - t0
    print(f"\n[done] {n_total} images in {dt:.1f}s "
          f"({n_total/max(dt,1):.2f} img/s)")
    if n_total:
        print(f"  offload rate     : {n_routed/n_total*100:.2f}%")
        print(f"  avg bandwidth    : {bw_sum/n_total:.2f} KB/sample")
        print(f"  avg latency      : {lat_sum/n_total:.1f} ms/sample")
    if n_with_label:
        print(f"  top-1 accuracy   : {n_correct/n_with_label*100:.2f}% "
              f"({n_correct}/{n_with_label})")
    print(f"  CSV              : {out}")


if __name__ == "__main__":
    main()

"""Build deploy/checkpoints.tar.gz from local artifacts.

Bundles the three things both servers need:
  - edge_v3l.pt              (MobileNetV3-Large checkpoint with 'state_dict' + 'arch')
  - qualitygate_fusion_3L.pt (cloud QualityGate head, paper-1 format)
  - class_names.txt          (89 class names, one per line, in label-id order)

Run this once on the dev machine (where the trained models live), then
upload the tarball to a release / private bucket and set
CHECKPOINTS_URL=<that-url> when running setup.sh on Pi/Jetson.

Usage:
    python deploy/package_checkpoints.py \\
        --edge-ckpt outputs/edge_model_v3l/best.pt \\
        --cloud-ckpt outputs/cloud_head/qualitygate_fusion_3L.pt \\
        --out deploy/checkpoints.tar.gz

If --edge-ckpt is missing 'arch' or 'state_dict' wrappers, this script
normalizes them so the deploy code works without conditional branches.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import torch


def _add_paper2_root_to_path() -> Path:
    """Make sure we can import the PlantWild loader for class names."""
    here = Path(__file__).resolve()
    for parent in [here.parents[1], here.parents[2], here.parents[3]]:
        if (parent / "paper1" / "src" / "datasets.py").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise SystemExit("could not locate TurboQuant root with paper1/src/datasets.py")


def normalize_edge(in_path: Path, out_path: Path, arch_default: str = "mobilenet_v3_large") -> None:
    ck = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    arch = ck.get("arch", arch_default)
    torch.save({"state_dict": sd, "arch": arch}, out_path)
    print(f"[edge] {in_path} -> {out_path} (arch={arch}, params={sum(v.numel() for v in sd.values())})")


def copy_cloud(in_path: Path, out_path: Path) -> None:
    ck = torch.load(in_path, map_location="cpu", weights_only=False)
    torch.save(ck, out_path)
    print(f"[cloud] {in_path} -> {out_path}")


def write_class_names(out_path: Path) -> int:
    _add_paper2_root_to_path()
    from paper1.src.datasets import load_plantwild  # type: ignore
    _, _, class_names = load_plantwild()
    out_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"[meta] {out_path} ({len(class_names)} classes)")
    return len(class_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-ckpt",  default="outputs/edge_model_v3l/best.pt")
    ap.add_argument("--cloud-ckpt", default="outputs/cloud_head/qualitygate_fusion_3L.pt")
    ap.add_argument("--out", default="deploy/checkpoints.tar.gz")
    args = ap.parse_args()

    edge_in = Path(args.edge_ckpt)
    cloud_in = Path(args.cloud_ckpt)
    if not edge_in.exists():
        raise SystemExit(f"missing edge ckpt: {edge_in}")
    if not cloud_in.exists():
        raise SystemExit(f"missing cloud ckpt: {cloud_in}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        normalize_edge(edge_in, td / "edge_v3l.pt")
        copy_cloud(cloud_in, td / "qualitygate_fusion_3L.pt")
        write_class_names(td / "class_names.txt")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:gz") as tar:
            for f in ("edge_v3l.pt", "qualitygate_fusion_3L.pt", "class_names.txt"):
                tar.add(td / f, arcname=f)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"[done] {out}  ({size_mb:.1f} MB)")
        print("Upload this tarball somewhere reachable, then on each device:")
        print(f"  CHECKPOINTS_URL=<url> bash deploy/setup.sh")


if __name__ == "__main__":
    main()

"""PlantWild split used throughout the paper (MVPDR protocol, seed 1).

The split is shipped as artifacts/plantwild_split.csv (relative image path,
label, split), exported from the loader the experiments were run with, in the
same order, so features, logits and predictions stay aligned. Images are not
redistributed: download PlantWild (Wei et al., 2024) and point PLANTWILD_ROOT
at its `images/` directory (default: data/PlantWild/images).
"""

from __future__ import annotations

import csv
import os
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = ROOT / "artifacts" / "plantwild_split.csv"
Datum = namedtuple("Datum", "impath label")


def images_root() -> Path:
    return Path(os.environ.get("PLANTWILD_ROOT", ROOT / "data" / "PlantWild" / "images"))


def load_split():
    """-> train, val, test as lists of (absolute image path, label); class names by label index."""
    base = images_root()
    parts = {"train": [], "val": [], "test": []}
    names = {}
    with SPLIT_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            parts[r["split"]].append((str(base / r["image"]), int(r["label"])))
            names[int(r["label"])] = r["class_name"]
    return parts["train"], parts["val"], parts["test"], [names[i] for i in range(len(names))]


def load_plantwild():
    """Same contract as the original loader: (train, test, class_names)."""
    train, _, test, names = load_split()
    return train, test, names


def materialize_test(out_dir: Path, copy: bool = False) -> int:
    """Lay the 3 677 test images out as <out_dir>/<class>/<image> (hard links, or copies), the layout the
    fleet emulator and the edge-fidelity check read."""
    _, _, test, names = load_split()
    n = 0
    for path, label in test:
        src = Path(path)
        dst = Path(out_dir) / names[label] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if copy:
            import shutil
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
        n += 1
    return n


class PlantWild:
    """Drop-in for the original PlantWild class (train_x / val / test / classnames)."""

    def __init__(self, root=None, num_shots: int = -1):
        if root is not None:
            os.environ.setdefault("PLANTWILD_ROOT", str(Path(root) / "images"))
        tr, va, te, self.classnames = load_split()
        self.train_x = [Datum(p, l) for p, l in tr]
        self.val = [Datum(p, l) for p, l in va]
        self.test = [Datum(p, l) for p, l in te]
        self.num_classes = len(self.classnames)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PlantWild split helper")
    ap.add_argument("--materialize-test", metavar="OUT_DIR", default=str(ROOT / "data" / "plantwild_test"),
                    help="write the test split as OUT_DIR/<class>/<image> (default data/plantwild_test)")
    ap.add_argument("--copy", action="store_true", help="copy files instead of hard-linking them")
    a = ap.parse_args()
    tr, va, te, names = load_split()
    print(f"split: train={len(tr)} val={len(va)} test={len(te)} classes={len(names)} (images under {images_root()})")
    print(f"materialized {materialize_test(Path(a.materialize_test), a.copy)} test images -> {a.materialize_test}")

"""Split the cross-platform preprocessing deviation into JPEG decoding and resampling.

Run the same command on the workstation and on the Nano (Python 3.6 compatible):
  python cloud/pillow_split.py --images <ref_images dir> --list <ref_images.json> --out <host>.npz
then compare the two archives with --compare:
  python cloud/pillow_split.py --compare ws.npz nano.npz
Stored per image: the decoded RGB array (uint8), and for each backbone the
preprocessed tensor of this host's decode and of the OTHER host's decode
(the other archive is passed with --other, so resampling can be run on
identical pixels).
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cloud import trt_common as tc  # noqa: E402


def run(args):
    names = [r["image"] for r in json.load(open(args.list))]
    other = np.load(args.other) if args.other else None
    out = {"pillow": np.array(Image.__version__ if hasattr(Image, "__version__") else __import__("PIL").__version__)}
    for i, n in enumerate(names):
        dec = np.asarray(Image.open(os.path.join(args.images, n)).convert("RGB"))
        out["dec_%d" % i] = dec
        srcs = {"own": dec}
        if other is not None:
            srcs["oth"] = other["dec_%d" % i]
        for tag, arr in srcs.items():
            for spec in tc.BACKBONES:
                out["%s_%s_%d" % (tag, spec["name"], i)] = tc.preprocess(
                    Image.fromarray(arr), spec["size"], spec["mean"], spec["std"])
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, "pillow", out["pillow"])


def compare(ws_path, nano_path):
    ws, nano = np.load(ws_path), np.load(nano_path)
    n = len([k for k in ws.files if k.startswith("dec_")])
    rep = {"pillow_workstation": str(ws["pillow"]), "pillow_nano": str(nano["pillow"]), "n_images": n,
           "decode_max_abs_diff_255": 0, "decode_pixels_differing_frac": [], "per_backbone": {}}
    for i in range(n):
        a, b = ws["dec_%d" % i].astype(int), nano["dec_%d" % i].astype(int)
        if a.shape != b.shape:
            raise SystemExit("decoded shapes differ for image %d" % i)
        rep["decode_max_abs_diff_255"] = int(max(rep["decode_max_abs_diff_255"], np.abs(a - b).max()))
        rep["decode_pixels_differing_frac"].append(float((np.abs(a - b).max(-1) > 0).mean()))
    rep["decode_pixels_differing_frac"] = float(np.mean(rep["decode_pixels_differing_frac"]))
    for spec in tc.BACKBONES:
        nm = spec["name"]
        tot, dec_only, res_only = [], [], []
        for i in range(n):
            ws_ws, nano_nano = ws["own_%s_%d" % (nm, i)], nano["own_%s_%d" % (nm, i)]
            ws_of_nano = ws["oth_%s_%d" % (nm, i)]      # workstation resampler, Nano-decoded pixels
            nano_of_ws = nano["oth_%s_%d" % (nm, i)]    # Nano resampler, workstation-decoded pixels
            tot.append(np.abs(ws_ws - nano_nano).max())
            dec_only.append(np.abs(ws_ws - ws_of_nano).max())   # same resampler, different decode
            res_only.append(np.abs(ws_ws - nano_of_ws).max())   # same pixels, different resampler
        rep["per_backbone"][nm] = {"total_max": float(max(tot)), "decode_only_max": float(max(dec_only)),
                                   "resample_only_max": float(max(res_only))}
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--list")
    ap.add_argument("--out")
    ap.add_argument("--other", default="")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    if a.compare:
        r = compare(*a.compare)
        if a.report:
            json.dump(r, open(a.report, "w"), indent=1)
    else:
        run(a)

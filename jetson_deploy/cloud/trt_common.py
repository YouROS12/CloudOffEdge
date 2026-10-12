"""Torch-free pieces of the cloud pipeline, shared by the desktop exporter
(cloud/export_onnx.py) and the Jetson TensorRT server (cloud/server_trt.py).

Must stay importable on the Jetson Nano's Python 3.6 (JetPack 4): no
`from __future__ import annotations`, no walrus, numpy + Pillow only.

  preprocess()       == torchvision Resize(bicubic) + CenterCrop + ToTensor
                        + Normalize, as used by cloud/server.py
  QualityGateNP      == cloud/server.py QualityGate.forward, in numpy
"""

import math

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Order matters: the head was trained on concat(dinov2, dinov3, clip).
BACKBONES = [
    {"name": "dinov2_vitl14", "size": 336, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "dim": 1024},
    {"name": "dinov3_vitl16", "size": 336, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "dim": 1024},
    {"name": "clip_vitl14",   "size": 224, "mean": CLIP_MEAN,     "std": CLIP_STD,     "dim": 768},
]


def preprocess(img, size, mean, std):
    """PIL RGB image -> float32 [1, 3, size, size], matching torchvision's
    Resize(size, BICUBIC) -> CenterCrop(size) -> ToTensor -> Normalize."""
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    img = img.resize((new_w, new_h), Image.BICUBIC)
    top = int(round((new_h - size) / 2.0))
    left = int(round((new_w - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0                 # [H, W, 3]
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])       # [1, 3, H, W]


def l2_normalize(x, axis=-1, eps=1e-12):
    n = np.sqrt((x * x).sum(axis=axis, keepdims=True))
    return x / np.maximum(n, eps)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class QualityGateNP(object):
    """numpy port of cloud/server.py QualityGate (weights from an .npz made
    by export_onnx.py)."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.W = d["linear_weight"].astype(np.float32)             # [C, D]
        self.b = d["linear_bias"].astype(np.float32)               # [C]
        self.protos_n = l2_normalize(d["prototypes"].astype(np.float32))  # [C, K, D]
        self.gate = 1.0 / (1.0 + np.exp(-d["gate_logits"].astype(np.float32)))  # [C]
        self.temp = float(math.exp(float(d["log_temp"])))

    def __call__(self, x):
        x_n = l2_normalize(x.astype(np.float32))                    # [B, D]
        linear_logits = x_n.dot(self.W.T) + self.b                  # [B, C]
        sims = np.einsum("bd,ckd->bck", x_n, self.protos_n)         # [B, C, K]
        proto_logits = sims.max(axis=-1) / self.temp                # [B, C]
        return self.gate * linear_logits + (1.0 - self.gate) * proto_logits


def fuse(feats):
    """List of raw [1, D_i] backbone outputs -> fused [1, sum D_i], each
    L2-normalized then the concat re-normalized (as in cloud/server.py)."""
    return l2_normalize(np.concatenate([l2_normalize(f.astype(np.float32)) for f in feats], axis=-1))


def load_class_names(path):
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]

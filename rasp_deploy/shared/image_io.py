"""Image encode/decode helpers shared by edge + cloud servers."""

from __future__ import annotations

import base64
import io
from typing import Tuple

from PIL import Image


def encode_jpeg(img: Image.Image, quality: int = 90) -> Tuple[bytes, float]:
    """Encode PIL image to JPEG bytes. Returns (bytes, kb_size)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    raw = buf.getvalue()
    return raw, len(raw) / 1024.0


def encode_b64(img: Image.Image, quality: int = 90) -> Tuple[str, float]:
    """Encode PIL image to base64 JPEG string. Returns (b64, kb_size_of_jpeg)."""
    raw, kb = encode_jpeg(img, quality=quality)
    return base64.b64encode(raw).decode("ascii"), kb


def escalation_payload(img: Image.Image, short_side: int) -> Image.Image:
    """Downscale an image so its shorter side equals `short_side` before it is
    uploaded to the cloud. Uses the same size rule and bicubic filter as the
    cloud's own first transform (torchvision Resize(short_side)), so for the
    336 px backbones the cloud-side resize becomes a no-op. Never upscales;
    short_side <= 0 returns the image unchanged."""
    if short_side <= 0 or min(img.size) <= short_side:
        return img
    w, h = img.size
    if w <= h:
        size = (short_side, int(short_side * h / w))
    else:
        size = (int(short_side * w / h), short_side)
    return img.resize(size, Image.BICUBIC)


def decode_b64(b64: str) -> Image.Image:
    """Decode base64 JPEG string to PIL image."""
    raw = base64.b64decode(b64.encode("ascii"))
    return Image.open(io.BytesIO(raw)).convert("RGB")

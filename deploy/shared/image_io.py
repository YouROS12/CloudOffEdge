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


def decode_b64(b64: str) -> Image.Image:
    """Decode base64 JPEG string to PIL image."""
    raw = base64.b64decode(b64.encode("ascii"))
    return Image.open(io.BytesIO(raw)).convert("RGB")

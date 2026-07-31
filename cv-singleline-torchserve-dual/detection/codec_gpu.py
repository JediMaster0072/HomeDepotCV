"""Lossless image decoding for detection requests."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def base64_png_to_numpy_image(image_base64: str) -> np.ndarray:
    if not isinstance(image_base64, str):
        raise TypeError(f"expected base64 string, got {type(image_base64)}")
    image_bytes = base64.b64decode(image_base64.encode("ascii"))
    with io.BytesIO(image_bytes) as buffer:
        image = Image.open(buffer)
        image.load()
    return np.array(image)

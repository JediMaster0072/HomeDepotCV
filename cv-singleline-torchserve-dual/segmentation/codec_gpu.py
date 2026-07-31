"""Lossless PNG encoding and decoding for segmentation requests and masks."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def numpy_image_to_base64_png(image: np.ndarray) -> str:
    image = np.asarray(image)
    if image.dtype == np.bool_:
        image = image.astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 or bool image, got {image.dtype}")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim not in (2, 3):
        raise ValueError(f"expected HxW or HxWxC image, got {image.shape}")
    with io.BytesIO() as buffer:
        Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_png_to_numpy_image(image_base64: str) -> np.ndarray:
    if not isinstance(image_base64, str):
        raise TypeError(f"expected base64 string, got {type(image_base64)}")
    image_bytes = base64.b64decode(image_base64.encode("ascii"))
    with io.BytesIO(image_bytes) as buffer:
        image = Image.open(buffer)
        image.load()
    return np.array(image)

"""Lossless image decoding for detection requests.

PIL decodes as RGB. Stage1 expects OpenCV BGR and converts BGR→RGB
internally (`img[:, :, ::-1]`), so request images are returned as BGR.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def _pil_array_to_bgr(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr.copy()
        arr[..., :3] = arr[..., :3][..., ::-1]
    return arr


def base64_png_to_numpy_image(image_base64: str) -> np.ndarray:
    if not isinstance(image_base64, str):
        raise TypeError(f"expected base64 string, got {type(image_base64)}")
    image_bytes = base64.b64decode(image_base64.encode("ascii"))
    with io.BytesIO(image_bytes) as buffer:
        image = Image.open(buffer)
        image.load()
    return _pil_array_to_bgr(image)

"""Crop preprocessing helpers for scripts under scripts/common/."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.crop_preprocess import (  # noqa: F401
    DEFAULT_MIN_CROP_SHORT_SIDE,
    DEFAULT_MIN_CROP_SHORT_SIDE_HD,
    DEFAULT_MIN_CROP_SHORT_SIDE_SD,
    DEFAULT_OCR_RETRY_ROTATION_ANGLES,
    DEFAULT_OCR_ROTATION_ANGLES,
    MAX_DESKEW_DEGREES,
    MIN_DESKEW_DEGREES,
    angle_slug,
    best_display_angle_from_rotation_hints,
    crop_short_side,
    enhance_crop_for_display,
    ensure_min_crop_resolution,
    prepare_label_crop_for_review,
    rotate_image_keep_bounds,
    sharpen_upscaled_crop,
    upscale_text_crop_multistep,
)


def crop_fullres_context(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    pad_px: int = 120,
    scale: float = 3.0,
    display_min_short_side: int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    from common.golden_shapes import expand_bbox

    image_h, image_w = image_rgb.shape[:2]
    context_bbox = expand_bbox(bbox, image_w, image_h, pad_px=pad_px, scale=scale)
    x1, y1, x2, y2 = context_bbox
    crop = image_rgb[y1:y2, x1:x2].copy()
    if display_min_short_side > 0:
        crop = ensure_min_crop_resolution(crop, min_short_side=display_min_short_side, sharpen=True)
    return crop, context_bbox

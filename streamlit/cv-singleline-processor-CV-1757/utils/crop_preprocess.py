"""Shared crop preprocessing helpers for golden-dataset tooling and OCR."""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_MIN_CROP_SHORT_SIDE_SD = 480
DEFAULT_MIN_CROP_SHORT_SIDE_HD = 720
DEFAULT_MIN_CROP_SHORT_SIDE = DEFAULT_MIN_CROP_SHORT_SIDE_HD

# Golden eval runs every angle; production OCR retries these after a weak upright read.
DEFAULT_OCR_ROTATION_ANGLES = (0.0, 180.0, -90.0, 90.0, -10.0, 10.0, -5.0, 5.0)
DEFAULT_OCR_RETRY_ROTATION_ANGLES = (180.0, -90.0, 90.0, -10.0, 10.0, -5.0, 5.0)
MIN_DESKEW_DEGREES = 2.0
MAX_DESKEW_DEGREES = 45.0


def crop_short_side(image: np.ndarray) -> int:
    if image is None or image.size == 0:
        return 0
    height, width = image.shape[:2]
    return min(height, width)


def _upscale_interpolation(source_short_side: int) -> int:
    """
    Legacy single-step interpolation picker.

    Prefer upscale_text_crop_multistep() for tiny label crops — a one-shot
    nearest-neighbor jump from ~50px to 720px looks very blocky in review UI.
    """
    if source_short_side < 80:
        return cv2.INTER_NEAREST
    if source_short_side < 160:
        return cv2.INTER_CUBIC
    return cv2.INTER_LANCZOS4


def sharpen_upscaled_crop(image: np.ndarray, *, amount: float = 1.6, sigma: float = 1.0) -> np.ndarray:
    """Light unsharp mask after upscaling to recover digit edges."""
    if image is None or image.size == 0:
        return image

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image, amount, blurred, 1.0 - amount, 0)
    return np.clip(sharpened, 0, 255).astype(image.dtype)


def upscale_text_crop_multistep(
    image: np.ndarray,
    target_short_side: int,
    *,
    max_step_scale: float = 2.0,
    prefilter_tiny: bool = True,
) -> np.ndarray:
    """
    Upscale label crops in 2x steps for smoother enlargement of tiny text.

    Tiny warehouse labels (~40–80px tall) upscaled in one jump look blocky.
    Stepwise Lanczos4 keeps digit strokes cleaner for human review and OCR.
    """
    if image is None or image.size == 0 or target_short_side <= 0:
        return image

    height, width = image.shape[:2]
    short_side = min(height, width)
    if short_side >= target_short_side:
        return image

    current = image
    if prefilter_tiny and short_side < 80:
        current = cv2.bilateralFilter(current, d=5, sigmaColor=50, sigmaSpace=50)

    cur_short = min(current.shape[:2])
    while cur_short < target_short_side:
        remaining = target_short_side / float(cur_short)
        step = min(max_step_scale, remaining)
        if step <= 1.001:
            break
        new_width = max(1, int(round(current.shape[1] * step)))
        new_height = max(1, int(round(current.shape[0] * step)))
        current = cv2.resize(current, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        cur_short = min(current.shape[:2])

    cur_short = min(current.shape[:2])
    if cur_short != target_short_side:
        scale = target_short_side / float(cur_short)
        new_width = max(1, int(round(current.shape[1] * scale)))
        new_height = max(1, int(round(current.shape[0] * scale)))
        current = cv2.resize(current, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return current


def ensure_min_crop_resolution(
    image: np.ndarray,
    min_short_side: int = DEFAULT_MIN_CROP_SHORT_SIDE,
    sharpen: bool = True,
    *,
    use_multistep: bool = True,
) -> np.ndarray:
    """
    Upscale small crops so the short side is at least min_short_side pixels.

    Uses stepwise Lanczos upscaling for tiny sources, then optional sharpen.
    """
    if image is None or image.size == 0 or min_short_side <= 0:
        return image

    height, width = image.shape[:2]
    short_side = min(height, width)

    if short_side >= min_short_side:
        return image

    if use_multistep:
        upscaled = upscale_text_crop_multistep(image, min_short_side)
    else:
        scale = min_short_side / float(short_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        interpolation = _upscale_interpolation(short_side)
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    if sharpen and short_side < 160:
        upscaled = sharpen_upscaled_crop(upscaled)
    return upscaled


def enhance_crop_for_display(image: np.ndarray, deblur: bool = False) -> np.ndarray:
    """High-contrast view for human review without OCR-style binarization."""
    if image is None or image.size == 0:
        return image

    if deblur:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    if image.ndim == 3:
        lab = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        for sigma, amount in ((1.0, 1.4), (2.5, 0.8)):
            blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma)
            l_channel = cv2.addWeighted(l_channel, amount, blurred, 1.0 - amount, 0)
        l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        enhanced = sharpen_upscaled_crop(enhanced)

    return sharpen_upscaled_crop(enhanced)


def _normalize_sku_digits(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def best_display_angle_from_rotation_hints(rotation_hints: str, preferred_sku: str = "") -> float:
    """
    Pick the OCR rotation angle that best matches the merged SKU hint.

    Used only to deskew the human review view — not shown to labelers.
    """
    if not rotation_hints:
        return 0.0

    preferred = _normalize_sku_digits(preferred_sku)
    entries: list[tuple[float, str]] = []
    for part in str(rotation_hints).split("|"):
        if ":" not in part:
            continue
        angle_str, sku = part.split(":", 1)
        try:
            entries.append((float(angle_str), sku))
        except ValueError:
            continue

    if not entries:
        return 0.0

    def angle_rank(angle: float) -> tuple[float, float]:
        wrapped = abs(((angle + 180.0) % 360.0) - 180.0)
        return wrapped, abs(angle)

    if preferred:
        matching = [angle for angle, sku in entries if _normalize_sku_digits(sku) == preferred]
        if matching:
            return min(matching, key=angle_rank)

    for angle, _ in entries:
        if angle == 0.0:
            return 0.0
    return min((angle for angle, _ in entries), key=angle_rank)


def prepare_label_crop_for_review(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    min_short_side: int = DEFAULT_MIN_CROP_SHORT_SIDE,
    deblur: bool = False,
    enhance: bool = True,
    correction_angle: float = 0.0,
) -> np.ndarray:
    """
    Extract a label crop from a full-res source image and prepare it for review.

    Prefer this over reading pre-saved crops when the source image is available —
    saved crops may have been generated with older upscaling settings.
    """
    if image_rgb is None or image_rgb.size == 0:
        return image_rgb

    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2].copy()
    crop = ensure_min_crop_resolution(crop, min_short_side=min_short_side, sharpen=True)
    if correction_angle:
        crop = rotate_image_keep_bounds(crop, float(correction_angle))
    if enhance:
        crop = enhance_crop_for_display(crop, deblur=deblur)
    return crop


def angle_slug(angle: float) -> str:
    if float(angle).is_integer():
        return str(int(angle)).replace("-", "neg")
    return str(angle).replace("-", "neg").replace(".", "p")


def rotate_image_keep_bounds(image: np.ndarray, angle: float, border_value=255) -> np.ndarray:
    """Rotate a crop in-plane, expanding the canvas so no corners are clipped."""
    if angle == 0:
        return image.copy()

    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderValue=border_value)

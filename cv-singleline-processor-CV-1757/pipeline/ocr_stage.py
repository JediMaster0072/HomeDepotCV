import base64
import json
import urllib.error
import urllib.request

import numpy as np
import cv2
from google.cloud import vision

from utils.crop_preprocess import (
    DEFAULT_MIN_CROP_SHORT_SIDE,
    DEFAULT_OCR_RETRY_ROTATION_ANGLES,
    MAX_DESKEW_DEGREES,
    MIN_DESKEW_DEGREES,
    angle_slug,
    ensure_min_crop_resolution,
    rotate_image_keep_bounds,
)
from services.image_processor import numpy_image_to_base64_png
from utils.google_ocr_utils import (
    deskew_rotation_for_baseline,
    estimate_skew_degrees_from_annotations,
    is_strong_sku_read,
    merge_multi_pass_sku_results,
    parse_google_ocr_words,
)


def _tag_ocr_results(results: list[dict], suffix: str) -> list[dict]:
    for result in results:
        base = result.get("source", "google_ocr_sku_parse")
        result["source"] = f"{base}_{suffix}" if suffix else base
    return results


def _map_rot180_results_to_original(results: list[dict], image_shape: tuple[int, ...]) -> list[dict]:
    """Map OCR boxes from a 180-degree rotated image back to original coordinates."""
    img_h, img_w = image_shape[:2]

    for result in results:
        bbox = result.get("bbox")

        if bbox is None:
            continue

        old_x1, old_y1, old_x2, old_y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        bbox.x1 = max(0, img_w - old_x2)
        bbox.y1 = max(0, img_h - old_y2)
        bbox.x2 = min(img_w, img_w - old_x1)
        bbox.y2 = min(img_h, img_h - old_y1)

    return results


def _parse_ocr_pass(image: np.ndarray, call_annotations_fn) -> tuple[list[dict], list | None]:
    annotations = call_annotations_fn(image)
    if not annotations:
        return [], None
    return parse_google_ocr_words(annotations), annotations


def _run_google_ocr_words_with_callable(
    image: np.ndarray,
    call_annotations_fn,
    retry_with_enhanced_image: bool = True,
    retry_with_rotation_angles: bool = True,
    rotation_retry_angles: tuple[float, ...] | list[float] = DEFAULT_OCR_RETRY_ROTATION_ANGLES,
    min_crop_short_side: int = DEFAULT_MIN_CROP_SHORT_SIDE,
) -> list[dict]:
    """
    Run Google OCR with automatic deskew and short-circuit rotation retries.

    Flow:
    1. Upscale + OCR upright (0°)
    2. If weak/no SKU → deskew using OCR polygon baseline, OCR again
    3. If still weak → high-contrast retry (same geometry)
    4. If still weak → OCR at 180°, ±90°, ±10°, ±5°
    5. Return merged best valid SKUs across passes
    """
    if image is None or image.size == 0:
        return []

    image = ensure_min_crop_resolution(image, min_short_side=min_crop_short_side)
    passes: list[list[dict]] = []

    upright_results, upright_annotations = _parse_ocr_pass(image, call_annotations_fn)
    if upright_results:
        passes.append(upright_results)

    merged = merge_multi_pass_sku_results(passes)
    if is_strong_sku_read(merged):
        return merged

    skew = estimate_skew_degrees_from_annotations(upright_annotations)
    if skew is not None:
        deskew_angle = deskew_rotation_for_baseline(skew)
        if MIN_DESKEW_DEGREES <= abs(deskew_angle) <= MAX_DESKEW_DEGREES:
            deskewed_image = rotate_image_keep_bounds(image, deskew_angle)
            deskew_results, _ = _parse_ocr_pass(deskewed_image, call_annotations_fn)
            if deskew_results:
                passes.append(_tag_ocr_results(deskew_results, f"deskew{angle_slug(deskew_angle)}"))
                merged = merge_multi_pass_sku_results(passes)
                if is_strong_sku_read(merged):
                    return merged

    if retry_with_enhanced_image:
        enhanced_image = enhance_image_for_ocr(image)
        if enhanced_image is not image:
            enhanced_results, _ = _parse_ocr_pass(enhanced_image, call_annotations_fn)
            if enhanced_results:
                passes.append(_tag_ocr_results(enhanced_results, "enhanced_retry"))
                merged = merge_multi_pass_sku_results(passes)
                if is_strong_sku_read(merged):
                    return merged

    if retry_with_rotation_angles:
        for angle in rotation_retry_angles:
            if float(angle) == 0.0:
                continue

            rotated_image = rotate_image_keep_bounds(image, float(angle))
            rotated_results, _ = _parse_ocr_pass(rotated_image, call_annotations_fn)
            if not rotated_results:
                continue

            if float(angle) == 180.0:
                rotated_results = _map_rot180_results_to_original(rotated_results, image.shape)

            passes.append(_tag_ocr_results(rotated_results, f"rot{angle_slug(angle)}_retry"))
            merged = merge_multi_pass_sku_results(passes)
            if is_strong_sku_read(merged):
                return merged

    return merged


# Pipeline order: 28.5
# Description: Builds a high-contrast OCR retry image when the normal masked-original OCR pass finds no SKU.
def enhance_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Create a same-size, high-contrast image for a second OCR attempt.

    This does not change coordinates because the image dimensions stay identical.
    It only changes pixel contrast so faint or low-contrast SKU digits are easier
    for Google OCR to read.
    """
    if image is None or image.size == 0:
        return image

    if image.ndim == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    thresholded = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    if image.ndim == 3:
        return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)

    return thresholded


# Pipeline order: 29
# Description: Runs Google OCR on the masked original image and returns parsed SKU candidates.
def run_google_ocr_words(
    gcv_client: vision.ImageAnnotatorClient,
    image: np.ndarray,
    retry_with_enhanced_image: bool = True,
    retry_with_rotation_angles: bool = True,
) -> list[dict]:
    """
    Run Google Vision OCR on an in-memory NumPy image and return validated SKU results.

    Returns:
    [
        {
            "text": "1004334515",
            "raw_text": "1004 334-515",
            "bbox": BoundingBox(...),
            "confidence": 1.0,
            "source": "google_ocr_sku_parse",
        },
        ...
    ]
    """
    return _run_google_ocr_words_with_callable(
        image=image,
        call_annotations_fn=lambda ocr_image: call_google_ocr_np(gcv_client, ocr_image),
        retry_with_enhanced_image=retry_with_enhanced_image,
        retry_with_rotation_angles=retry_with_rotation_angles,
    )


def run_google_ocr_words_with_api_key(
    api_key: str,
    image: np.ndarray,
    retry_with_enhanced_image: bool = True,
    retry_with_rotation_angles: bool = True,
) -> list[dict]:
    """
    Run Google Vision OCR through the REST API-key flow.

    This is useful for local testing when ADC/service-account credentials are
    unavailable, but an approved Vision API key exists in the environment.
    """
    return _run_google_ocr_words_with_callable(
        image=image,
        call_annotations_fn=lambda ocr_image: call_google_ocr_api_key(api_key, ocr_image),
        retry_with_enhanced_image=retry_with_enhanced_image,
        retry_with_rotation_angles=retry_with_rotation_angles,
    )


# Pipeline order: 30
# Description: Encodes a NumPy image and calls Google Vision document text detection.
def call_google_ocr_np(gcv_client: vision.ImageAnnotatorClient, image: np.ndarray) -> list[dict] | None:
    """
    Calls Google Vision OCR using an in-memory OpenCV / NumPy image.

    image: np.ndarray in OpenCV BGR format
    """
    if image is None or image.size == 0:
        return None

    _, png_image_bytes = numpy_image_to_base64_png(image, return_bytes=True)

    gcv_image = vision.Image(content=png_image_bytes)

    response = gcv_client.document_text_detection(
        image=gcv_image,
        image_context=vision.ImageContext(language_hints=["en-t-i0"]),
    )

    if response.error.message:
        raise RuntimeError(f"Google Vision OCR error: {response.error.message}")

    return list(response.text_annotations)


def call_google_ocr_api_key(api_key: str, image: np.ndarray, timeout: int = 30) -> list[dict] | None:
    """
    Calls Google Vision OCR using a REST API key and returns textAnnotations.

    The key must be provided by the caller from an environment variable or local
    untracked config file. Do not hardcode keys in this repo.
    """
    if image is None or image.size == 0:
        return None

    if not api_key:
        raise ValueError("Google OCR API key is empty")

    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        raise ValueError("Failed to encode image for Google OCR API")

    img_base64 = base64.b64encode(buffer).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["en"]},
            }
        ]
    }
    request = urllib.request.Request(
        url=f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google OCR API-key request failed: HTTP {exc.code}: {body}") from exc

    response_payload = (data.get("responses") or [{}])[0]

    if "error" in response_payload:
        raise RuntimeError(f"Google OCR API-key response error: {response_payload['error']}")

    return response_payload.get("textAnnotations", [])

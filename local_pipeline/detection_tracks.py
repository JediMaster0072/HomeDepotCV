import logging

import numpy as np

from common_config import ClipTrack, NO_CROPS_DETECTED

logger = logging.getLogger(__name__)


def predict_local_detections(od_endpoint, raw_image: np.ndarray, encode_image_fn):
    """Run local YOLO detection and return the original fallback status on failure/no detections."""
    try:
        print("First numpy image", raw_image.sum(), raw_image.shape, raw_image.dtype)
        encoded_image = encode_image_fn(raw_image)
        detections = od_endpoint.predict_detection(encoded_image)
    except Exception as exc:
        logger.error("OD endpoint unreachable: %s - falling back to GCF.", exc)
        detections = []

    if not detections:
        logger.warning("No detections - full-image Google OCR fallback.")
        return NO_CROPS_DETECTED, []

    return None, detections


def build_detection_tracks(
    raw_image: np.ndarray,
    detections,
    bbox_buffer_pct: float,
    crop_fn,
    preprocess_fn,
    skip_class_ids: tuple[int, ...] = (4,),
) -> tuple[list[np.ndarray], list[ClipTrack]]:
    """Convert local detections into buffered, preprocessed crops and ClipTrack metadata."""
    img_h, img_w = raw_image.shape[:2]
    crops = []
    tracked_clips: list[ClipTrack] = []

    for det_id, det in enumerate(detections):
        if int(det.class_id) in skip_class_ids:
            continue

        buffered_det = det.apply_buffer(img_w, img_h, pct=bbox_buffer_pct)
        print("BOX's ->", det)

        crop = crop_fn(raw_image, buffered_det)
        crop = preprocess_fn(crop)

        if crop.size == 0:
            continue

        h, w = crop.shape[:2]
        track = ClipTrack(
            det_id=det_id,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            orig_bbox=det,
            buffered_bbox=buffered_det,
            clip_h=h,
            clip_w=w,
        )

        tracked_clips.append(track)
        crops.append(crop)

    return crops, tracked_clips


def build_clip_items(crops: list[np.ndarray], tracked_clips: list[ClipTrack]) -> list[dict]:
    """Package crops and tracks into the strip-creation input shape."""
    return [{"track": track, "crop": crop} for track, crop in zip(tracked_clips, crops)]

import numpy as np

from common_config import (
    ClipTrack,
    CROPS_ENDPOINT_FAILED,
    CROPS_NOT_DETECTED,
)


# Pipeline order: 14
# Description: Calls the detection endpoint and returns a fallback status when crops cannot be detected.
def predict_label_detections(vertex_endpoint, raw_image: np.ndarray, file_name: str, store_number: str):
    try:
        detections = vertex_endpoint.predict_detection(raw_image, file_name, store_number)
    except Exception as exc:
        print(f"OD endpoint unreachable: {exc} – falling back to original codebase.")
        return CROPS_ENDPOINT_FAILED, []

    if not len(detections):
        print("No detections – full-image Google OCR fallback.")
        return CROPS_NOT_DETECTED, []

    return None, detections


# Pipeline order: 17
# Description: Converts detections into preprocessed crops and ClipTrack objects used by later stages.
def build_detection_tracks(
    raw_image: np.ndarray,
    detections,
    bbox_buffer_pct: float,
    crop_fn,
    preprocess_fn,
) -> tuple[list[np.ndarray], list[ClipTrack]]:
    img_h, img_w = raw_image.shape[:2]
    crops = []
    tracked_clips: list[ClipTrack] = []

    for det_id, det in enumerate(detections):
        buffered_det = det.apply_buffer(img_w, img_h, pct=bbox_buffer_pct)

        crop = crop_fn(raw_image, buffered_det)
        crop = preprocess_fn(crop)

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


# Pipeline order: 19
# Description: Packages crops and tracks into the group input structure expected by strip creation.
def build_clip_items(crops: list[np.ndarray], tracked_clips: list[ClipTrack]) -> list[dict]:
    return [{"track": track, "crop": crop} for track, crop in zip(tracked_clips, crops)]

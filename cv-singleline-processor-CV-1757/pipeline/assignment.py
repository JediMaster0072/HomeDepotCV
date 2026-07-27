from common_config import BoundingBox, ClipTrack


# Pipeline order: Shared helper
# Description: Computes the center point of a bounding box.
def bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


# Pipeline order: Shared helper
# Description: Checks whether a point lies inside a bounding box.
def point_in_bbox(x: float, y: float, bbox: BoundingBox) -> bool:
    return bbox.x1 <= x < bbox.x2 and bbox.y1 <= y < bbox.y2


# Pipeline order: 35
# Description: Assigns an original-coordinate OCR word bbox back to the detected label track that most likely produced it.
def assign_word_to_track_in_original(
    word_bbox: BoundingBox,
    all_tracks: list,
) -> ClipTrack | None:
    """
    Assign an OCR word (in original-image coordinates) to the track whose
    buffered bounding box contains the word's centre.

    Falls back to the track with the highest overlap (IoU) if no track strictly
    contains the centre point, so partial matches near crop edges are still captured.
    """
    cx = (word_bbox.x1 + word_bbox.x2) / 2.0
    cy = (word_bbox.y1 + word_bbox.y2) / 2.0

    # 1. Strict containment check.
    for track in all_tracks:
        if track.buffered_bbox is None:
            continue
        bb = track.buffered_bbox
        if bb.x1 <= cx < bb.x2 and bb.y1 <= cy < bb.y2:
            return track

    # 2. IoU fallback: pick the track with the highest overlap.
    best_track = None
    best_iou = 0.0

    wx1, wy1, wx2, wy2 = word_bbox.x1, word_bbox.y1, word_bbox.x2, word_bbox.y2
    word_area = max(0, wx2 - wx1) * max(0, wy2 - wy1)

    if word_area == 0:
        return None

    for track in all_tracks:
        if track.buffered_bbox is None:
            continue
        bb = track.buffered_bbox
        ix1 = max(wx1, bb.x1)
        iy1 = max(wy1, bb.y1)
        ix2 = min(wx2, bb.x2)
        iy2 = min(wy2, bb.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            continue
        bb_area = max(0, bb.x2 - bb.x1) * max(0, bb.y2 - bb.y1)
        iou = inter / (word_area + bb_area - inter)
        if iou > best_iou:
            best_iou = iou
            best_track = track

    return best_track if best_iou > 0 else None

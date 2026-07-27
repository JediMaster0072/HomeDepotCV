from common_config import BoundingBox


def bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    """Return the center point of a bounding box."""
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def point_in_bbox(x: float, y: float, bbox: BoundingBox) -> bool:
    """Check whether a point lies inside a bounding box."""
    return bbox.x1 <= x < bbox.x2 and bbox.y1 <= y < bbox.y2


def assign_word_to_track(word_bbox: BoundingBox, group):
    """Assign a strip-local OCR word to the track whose crop contains its center."""
    cx, cy = bbox_center(word_bbox)

    for item in group:
        track = item["track"]
        if track.crop_strip_bbox and point_in_bbox(cx, cy, track.crop_strip_bbox):
            return track

    return None


def map_strip_word_to_original(word_bbox: BoundingBox, track) -> BoundingBox:
    """Map OCR word coordinates from strip space back to original image coordinates."""
    if track.strip_bbox is None:
        raise ValueError("track.strip_bbox missing")
    if track.buffered_bbox is None:
        raise ValueError("track.buffered_bbox missing")

    strip_x1 = track.strip_bbox.x1
    strip_y1 = track.strip_bbox.y1
    pad_x = track.pad_offset_x or 0
    pad_y = track.pad_offset_y or 0
    crop_orig_x1 = track.buffered_bbox.x1
    crop_orig_y1 = track.buffered_bbox.y1

    return BoundingBox(
        x1=int(word_bbox.x1 - strip_x1 - pad_x + crop_orig_x1),
        y1=int(word_bbox.y1 - strip_y1 - pad_y + crop_orig_y1),
        x2=int(word_bbox.x2 - strip_x1 - pad_x + crop_orig_x1),
        y2=int(word_bbox.y2 - strip_y1 - pad_y + crop_orig_y1),
        class_id=track.class_id,
        class_name=track.class_name,
    )


def assign_word_to_track_in_original(word_bbox: BoundingBox, all_tracks: list):
    """Assign an original-coordinate OCR bbox to the best matching detected track."""
    cx = (word_bbox.x1 + word_bbox.x2) / 2.0
    cy = (word_bbox.y1 + word_bbox.y2) / 2.0

    for track in all_tracks:
        if track.buffered_bbox is None:
            continue
        bb = track.buffered_bbox
        if bb.x1 <= cx < bb.x2 and bb.y1 <= cy < bb.y2:
            return track

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

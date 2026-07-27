import numpy as np

from common_config import BoundingBox


# Pipeline order: 20.1
# Description: Places one crop in the center of a fixed-size padded canvas.
def center_pad(img, target_h, target_w, pad_value=0):
    h, w = img.shape[:2]
    padded = np.full((target_h, target_w, 3), pad_value, dtype=img.dtype)

    y_offset = (target_h - h) // 2
    x_offset = (target_w - w) // 2

    padded[y_offset : y_offset + h, x_offset : x_offset + w] = img
    return padded, x_offset, y_offset


# Pipeline order: 19
# Description: Splits detected crop items into small groups that become segmentation strips.
def group_clip_items(items, max_per_strip=5):
    return [items[i : i + max_per_strip] for i in range(0, len(items), max_per_strip)]


# Pipeline order: 20
# Description: Builds one horizontal segmentation strip and records where each crop sits inside it.
def create_strip(group, strip_index: int) -> np.ndarray:
    """
    Build one segmentation strip exactly like training:
    - take only the clips in this group
    - compute group-local max_h and max_w
    - center-pad every clip to that group-local size
    - concatenate padded clips horizontally

    group: list[{"track": ClipTrack, "crop": np.ndarray}]

    Side effects:
    - updates each track with pad metadata
    - updates each track with strip-local padded slot bbox
    - updates each track with strip-local actual crop bbox
    """
    if not group:
        raise ValueError("Empty group passed to create_strip")

    # Group-local padding dimensions. This is the important fix.
    group_max_h = max(item["crop"].shape[0] for item in group)
    group_max_w = max(item["crop"].shape[1] for item in group)

    strip_h = group_max_h
    strip_w = group_max_w * len(group)
    dtype = group[0]["crop"].dtype

    strip = np.zeros((strip_h, strip_w, 3), dtype=dtype)

    cursor_x = 0
    for slot_idx, item in enumerate(group):
        track = item["track"]
        crop = item["crop"]

        padded, x_offset, y_offset = center_pad(
            crop,
            target_h=group_max_h,
            target_w=group_max_w,
            pad_value=0,
        )

        # Store padded crop for debugging / later inspection if needed.
        item["padded_crop"] = padded

        # Pad metadata must be group-local, not global.
        track.pad_h = group_max_h
        track.pad_w = group_max_w
        track.pad_offset_x = x_offset
        track.pad_offset_y = y_offset

        strip[:, cursor_x : cursor_x + group_max_w] = padded

        track.strip_index = strip_index
        track.strip_slot = slot_idx

        # Full padded slot occupied by this crop in the strip.
        track.strip_bbox = BoundingBox(
            x1=cursor_x,
            y1=0,
            x2=cursor_x + group_max_w,
            y2=group_max_h,
            confidence=track.confidence,
            class_id=track.class_id,
            class_name=track.class_name,
        )

        # Actual crop region inside the strip after center-padding.
        track.crop_strip_bbox = BoundingBox(
            x1=cursor_x + x_offset,
            y1=y_offset,
            x2=cursor_x + x_offset + (track.clip_w or crop.shape[1]),
            y2=y_offset + (track.clip_h or crop.shape[0]),
            confidence=track.confidence,
            class_id=track.class_id,
            class_name=track.class_name,
        )

        cursor_x += group_max_w

    return strip

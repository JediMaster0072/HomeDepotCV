import cv2
import numpy as np


def mask_single_strip(
    seg_endpoint,
    strip: np.ndarray,
    group,
    encode_image_fn,
    decode_image_fn,
    use_segmentation: bool,
    debug_save_fn=None,
    current_image_name: str = "",
    strip_name: str = "strip",
) -> np.ndarray:
    """Run local segmentation for one strip and preserve clips without usable masks."""
    encoded_strip = encode_image_fn(strip)
    mask, strip_array, orig_mask = predict_binary_strip_mask(
        seg_endpoint=seg_endpoint,
        strip=encoded_strip,
        decode_image_fn=decode_image_fn,
        use_segmentation=use_segmentation,
    )

    if debug_save_fn and use_segmentation:
        debug_save_fn(
            mask if mask is not None else np.zeros(strip_array.shape[:2], dtype=np.uint8),
            "strips_viz",
            f"{strip_name}_pred_mask.jpg",
            current_image_name,
        )
        debug_save_fn(
            orig_mask if orig_mask is not None else np.zeros(strip_array.shape[:2], dtype=np.uint8),
            "strips_viz_orig_mask",
            f"{strip_name}_pred_mask.jpg",
            current_image_name,
        )

    if mask is None or mask.sum() == 0:
        for item in group:
            item["track"].seg_found = False
        masked_strip = strip_array.copy()
    else:
        masked_strip = apply_mask_to_strip_preserve_unsegmented_clips(
            strip=strip_array,
            mask=mask,
            group=group,
        )

    if debug_save_fn:
        debug_save_fn(masked_strip, "rois", f"{strip_name}_masked.jpg", current_image_name)

    return masked_strip


def predict_binary_strip_mask(seg_endpoint, strip: str, decode_image_fn, use_segmentation: bool):
    """Call the local segmentation endpoint and return mask, decoded strip, and original mask."""
    if use_segmentation:
        seg_output, strip_array, orig_mask = seg_endpoint.predict_segmentation(strip)
    else:
        seg_output, strip_array, orig_mask = None, decode_image_fn(strip), None

    mask = seg_output if isinstance(seg_output, np.ndarray) else None

    if use_segmentation and mask is not None and mask.shape[:2] != strip_array.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match strip shape {strip_array.shape[:2]}")

    return mask, strip_array, orig_mask


def apply_mask_to_strip_preserve_unsegmented_clips(
    strip: np.ndarray,
    mask: np.ndarray | None,
    group,
) -> np.ndarray:
    """Apply segmentation only to crop regions where the crop has a usable mask."""
    output = strip.copy()

    for item in group:
        track = item["track"]

        if track.crop_strip_bbox is None:
            continue

        box = track.crop_strip_bbox
        h, w = strip.shape[:2]
        x1 = max(0, box.x1)
        y1 = max(0, box.y1)
        x2 = min(w, box.x2)
        y2 = min(h, box.y2)

        if x2 <= x1 or y2 <= y1:
            continue

        has_seg = clip_has_segmentation(mask, track)

        if has_seg:
            track.seg_found = True
            clip = strip[y1:y2, x1:x2]
            clip_mask = mask[y1:y2, x1:x2]
            masked_clip = cv2.bitwise_and(clip, clip, mask=clip_mask)
            output[y1:y2, x1:x2] = masked_clip
            item["masked_strip_clip"] = masked_clip
        else:
            track.seg_found = False
            output[y1:y2, x1:x2] = strip[y1:y2, x1:x2]
            item["masked_strip_clip"] = None

    return output


def clip_has_segmentation(
    mask: np.ndarray,
    track,
    min_white_pixels: int = 40,
    min_white_ratio: float = 0.001,
) -> bool:
    """Decide whether one crop has enough segmentation mask pixels to trust."""
    if mask is None or track.crop_strip_bbox is None:
        return False

    box = track.crop_strip_bbox
    h, w = mask.shape[:2]
    x1 = max(0, box.x1)
    y1 = max(0, box.y1)
    x2 = min(w, box.x2)
    y2 = min(h, box.y2)

    if x2 <= x1 or y2 <= y1:
        return False

    clip_mask = mask[y1:y2, x1:x2]
    white_pixels = int(np.count_nonzero(clip_mask > 0))
    total_pixels = clip_mask.shape[0] * clip_mask.shape[1]

    if total_pixels == 0:
        return False

    white_ratio = white_pixels / total_pixels
    return white_pixels >= min_white_pixels and white_ratio >= min_white_ratio


def apply_mask_to_strip(strip: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask directly to a whole strip."""
    return cv2.bitwise_and(strip, strip, mask=mask)

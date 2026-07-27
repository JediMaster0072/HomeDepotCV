import cv2
import numpy as np

from common_config import ClipTrack, SEG_ENDPOINT_FAILED


# Pipeline order: 21
# Description: Runs segmentation for one strip and stores per-crop masked-strip clips for later original-image reconstruction.
def mask_single_strip(
    vertex_endpoint,
    strip: np.ndarray,
    group,
    file_name: str,
    store_number: str,
) -> np.ndarray:
    # 1. get one binary mask for the whole strip
    mask = predict_binary_strip_mask(vertex_endpoint, strip, file_name=file_name, store_number=store_number)

    # If no mask at all, preserve the full strip and still run OCR.
    # This means no clip is blacked out.
    if mask is None or mask.sum() == 0:
        for item in group:
            item["track"].seg_found = False

        masked_strip = strip.copy()

    else:
        mask = process_binary_mask_with_rotation(mask)

        # Apply mask only to clips where segmentation exists.
        # Preserve original image region for clips without segmentation.
        masked_strip = apply_mask_to_strip_preserve_unsegmented_clips(
            strip=strip,
            mask=mask,
            group=group,
        )

    return masked_strip


# Pipeline order: 22
# Description: Calls the segmentation endpoint and normalizes its response into a binary strip mask.
def predict_binary_strip_mask(vertex_endpoint, strip: np.ndarray, file_name: str = "", store_number: str = "") -> np.ndarray | None:
    """
    Expected segmentation output:
    - a single binary mask of shape [H, W]
    - or something convertible into that
    """
    try:
        seg_output = vertex_endpoint.predict_segmentation(strip, file_name, store_number)
    except Exception as exc:
        status = SEG_ENDPOINT_FAILED
        print(f"{status} - SEG endpoint unreachable for {file_name} | {store_number}: {exc} – Continuing with no segmentation results.")
        return None

    if seg_output is None:
        print(f"parse_segmentation_outputs returned None for {file_name} | {store_number}")
        return None

    if isinstance(seg_output, np.ndarray):
        print(f"Successfully obtained segmentation mask with shape {seg_output.shape} for {file_name}")
        mask = seg_output
    else:
        print(f"Unexpected return type from parse_segmentation_outputs: {type(seg_output)}. Expected np.ndarray or None.")
        return None

    try:
        if mask.ndim == 3:
            mask = np.squeeze(mask)

        if mask.shape[:2] != strip.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape[:2]} does not match strip shape {strip.shape[:2]} for {file_name} | {store_number}.")

        mask = (mask > 0).astype(np.uint8) * 255
        return mask
    except ValueError:
        raise
    except Exception as e:
        print(f"Error processing segmentation mask: {type(e).__name__}: {e}")
        return None


# Pipeline order: 26
# Description: Expands rotated segmentation mask regions so OCR keeps enough character context.
def process_binary_mask_with_rotation(binary_mask: np.ndarray, masks_np=None, scale_factor_w=1.5, scale_factor_h=1.35) -> np.ndarray:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        # Increase box size : W 50%, H 35%
        w *= scale_factor_w
        h *= scale_factor_h

        enlarged_rect = ((cx, cy), (w, h), angle)

        # Clipping after boxPoints is correct for any rotation angle.
        img_h, img_w = binary_mask.shape[:2]
        box = cv2.boxPoints(enlarged_rect)
        box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
        box = box.astype(np.int32)

        binary_mask = cv2.drawContours(binary_mask, [box], 0, 255, cv2.FILLED)

    return binary_mask


# Pipeline order: 27
# Description: Applies segmentation masks to crops with valid masks and stores masked clips for full-image reconstruction.
def apply_mask_to_strip_preserve_unsegmented_clips(
    strip: np.ndarray,
    mask: np.ndarray | None,
    group,
) -> np.ndarray:
    """
    Apply segmentation mask only for clips where segmentation exists.

    For clips without segmentation, preserve the original clip region.
    """
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

            # Store masked clip so _build_masked_original_image can use it
            # to paint ROI pixels from the original image at full resolution.
            item["masked_strip_clip"] = masked_clip

        else:
            track.seg_found = False
            output[y1:y2, x1:x2] = strip[y1:y2, x1:x2]
            item["masked_strip_clip"] = None

    return output


# Pipeline order: 27.1
# Description: Determines whether one crop has enough mask pixels to trust segmentation for that crop.
def clip_has_segmentation(
    mask: np.ndarray,
    track: ClipTrack,
    min_white_pixels: int = 20,
    min_white_ratio: float = 0.001,
) -> bool:
    """
    Decide whether this clip has a valid segmentation region.

    Checks white pixels inside the actual crop region within the strip,
    not the full padded slot.
    """
    if track.crop_strip_bbox is None:
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


# Pipeline order: Optional helper
# Description: Applies a binary mask directly to a strip without per-crop fallback logic.
def apply_mask_to_strip(strip: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(strip, strip, mask=mask)

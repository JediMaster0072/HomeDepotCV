import cv2
import numpy as np


# Pipeline order: 28
# Description: Reconstructs a full-resolution image where only segmented SKU regions remain visible for OCR.
def build_masked_original_image(
    raw_image: np.ndarray,
    all_groups: list,
    background_mode: str,
    mask_roi_pad_px: int,
    small_label_area_threshold: int,
    blur_sigma: int,
) -> np.ndarray:
    """
    Build a version of the original image where only segmented SKU ROI regions
    are visible; everything else is replaced by the configured background.
    """
    img_h, img_w = raw_image.shape[:2]

    if background_mode == "white":
        canvas = np.full_like(raw_image, 255)
    elif background_mode == "blur":
        k = blur_sigma | 1
        canvas = cv2.GaussianBlur(raw_image, (k, k), blur_sigma)
    else:
        canvas = np.zeros_like(raw_image)

    if mask_roi_pad_px > 0:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * mask_roi_pad_px + 1, 2 * mask_roi_pad_px + 1))
    else:
        dilation_kernel = None

    # Paint larger detections first so smaller, more precise detections can win
    # if their buffered regions overlap.
    all_items = [item for group in all_groups for item in group]
    all_items.sort(
        key=lambda it: (
            (it["track"].buffered_bbox.x2 - it["track"].buffered_bbox.x1) * (it["track"].buffered_bbox.y2 - it["track"].buffered_bbox.y1)
            if it["track"].buffered_bbox
            else 0
        ),
        reverse=True,
    )

    for item in all_items:
        track = item["track"]

        if track.buffered_bbox is None:
            continue

        bb = track.buffered_bbox
        orig_x1 = max(0, bb.x1)
        orig_y1 = max(0, bb.y1)
        orig_x2 = min(img_w, bb.x2)
        orig_y2 = min(img_h, bb.y2)

        if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
            continue

        crop_area = (orig_x2 - orig_x1) * (orig_y2 - orig_y1)
        is_small = small_label_area_threshold > 0 and crop_area < small_label_area_threshold

        if is_small or not track.seg_found or track.crop_strip_bbox is None:
            canvas[orig_y1:orig_y2, orig_x1:orig_x2] = raw_image[orig_y1:orig_y2, orig_x1:orig_x2]
            continue

        masked_clip = item.get("masked_strip_clip", None)

        if masked_clip is not None and masked_clip.size > 0:
            crop_h = orig_y2 - orig_y1
            crop_w = orig_x2 - orig_x1

            resized = cv2.resize(
                masked_clip,
                (crop_w, crop_h),
                interpolation=cv2.INTER_NEAREST,
            )

            if resized.ndim == 3:
                clip_bin = (resized.sum(axis=2) > 0).astype(np.uint8)
            else:
                clip_bin = (resized > 0).astype(np.uint8)

            if dilation_kernel is not None:
                clip_bin = cv2.dilate(clip_bin, dilation_kernel)
                clip_bin = clip_bin[:crop_h, :crop_w]

            orig_crop = raw_image[orig_y1:orig_y2, orig_x1:orig_x2]
            bg_crop = canvas[orig_y1:orig_y2, orig_x1:orig_x2]

            if orig_crop.ndim == 3:
                mask_3ch = np.stack([clip_bin] * orig_crop.shape[2], axis=-1)
                inv_3ch = 1 - mask_3ch
                canvas[orig_y1:orig_y2, orig_x1:orig_x2] = orig_crop * mask_3ch + bg_crop * inv_3ch
            else:
                canvas[orig_y1:orig_y2, orig_x1:orig_x2] = orig_crop * clip_bin + bg_crop * (1 - clip_bin)
        else:
            canvas[orig_y1:orig_y2, orig_x1:orig_x2] = raw_image[orig_y1:orig_y2, orig_x1:orig_x2]

    return canvas

from pathlib import Path

import cv2


def save_debug_image(img, debug: bool, debug_dir: Path, current_image_name: str, subdir, name, image_name=None):
    """Save an intermediate local pipeline image under the per-image debug folder."""
    if not debug:
        return

    if image_name:
        image_name = current_image_name.split(".")[0]

    path = debug_dir / image_name / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cv2.imwrite(str(path), img[:, :, ::-1])
    except IndexError:
        cv2.imwrite(str(path), img)


def save_ocr_results_on_original_image(raw_image, result, debug_dir: Path, file_path: str = "", output_name: str = "ocr_original_bboxes.jpg"):
    """Draw final OCR SKU boxes on the original image and save the local debug image."""
    if raw_image is None or raw_image.size == 0:
        return None

    if result is None or not result.raw_ocr_results:
        return None

    image_name = Path(file_path).stem if file_path else "in_memory_image"
    out_dir = debug_dir / image_name
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = raw_image.copy()
    img_h, img_w = vis.shape[:2]

    for item in result.raw_ocr_results:
        box = item.original_bbox
        x1 = max(0, min(img_w - 1, int(box.x1)))
        y1 = max(0, min(img_h - 1, int(box.y1)))
        x2 = max(0, min(img_w - 1, int(box.x2)))
        y2 = max(0, min(img_h - 1, int(box.y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        label = f"{item.text}"
        if item.class_name:
            label = f"{item.text} | {item.class_name}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_x1 = x1
        label_y1 = max(0, y1 - text_h - baseline - 8)
        label_x2 = min(img_w - 1, x1 + text_w + 8)
        label_y2 = min(img_h - 1, label_y1 + text_h + baseline + 8)

        cv2.rectangle(vis, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
        cv2.putText(
            vis,
            label,
            (label_x1 + 4, label_y2 - baseline - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    out_path = out_dir / output_name
    ok = cv2.imwrite(str(out_path), vis)

    if not ok:
        print(f"Failed to save OCR debug image: {out_path}")
        return None

    print(f"Saved OCR debug image: {out_path}")
    return out_path

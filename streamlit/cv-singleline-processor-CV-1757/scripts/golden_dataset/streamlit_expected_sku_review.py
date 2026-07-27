"""
Streamlit app for human review of expected_sku ground truth.

Usage:
  pip install streamlit pandas pillow
  streamlit run scripts/golden_dataset/streamlit_expected_sku_review.py

After review, sync labels into JSON:
  python scripts/golden_dataset/sync_expected_sku_from_truth_csv.py

Then measure OCR accuracy:
  python scripts/golden_dataset/run_golden_dataset_local.py --mode ocr-crops --save-crops
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.crop_preprocess import (
    best_display_angle_from_rotation_hints,
    crop_fullres_context,
    enhance_crop_for_display,
    prepare_label_crop_for_review,
)
from common.paths import path_for_csv, resolve_project_data_path, setup_script_paths
from common.sku_review import is_reviewed_expected_sku, parse_expected_sku_input

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)
DEFAULT_TRUTH_CSV = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "golden_sku_truth.csv"
DEFAULT_BATCH_IMAGES = (
    PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "review_batch_images.txt"
)
DEFAULT_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"


def load_rows(truth_csv: Path) -> list[dict]:
    with truth_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(truth_csv: Path, rows: list[dict], fieldnames: list[str]) -> None:
    for row in rows:
        for key in ("crop_path", "overlay_path"):
            raw = str(row.get(key, "") or "").strip()
            if not raw:
                continue
            resolved = resolve_project_data_path(raw, PROJECT_ROOT)
            if resolved.is_file():
                row[key] = path_for_csv(resolved, PROJECT_ROOT)
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reviewed_count(rows: list[dict]) -> int:
    return sum(1 for row in rows if is_reviewed_expected_sku(row.get("expected_sku", "")))


def load_batch_images(batch_file: Path) -> set[str]:
    if not batch_file.exists():
        return set()
    return {
        line.strip()
        for line in batch_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def filter_rows(
    rows: list[dict],
    label: str,
    image: str,
    only_unreviewed: bool,
    batch_images: set[str] | None = None,
) -> list[dict]:
    filtered = rows

    if batch_images:
        filtered = [row for row in filtered if row.get("image") in batch_images]

    if label and label != "All":
        filtered = [row for row in filtered if row.get("label") == label]

    if image and image != "All":
        filtered = [row for row in filtered if row.get("image") == image]

    if only_unreviewed:
        filtered = [
            row for row in filtered if not is_reviewed_expected_sku(row.get("expected_sku", ""))
        ]

    return filtered


def row_key(row: dict) -> str:
    if row.get("region_key"):
        return row["region_key"]
    return f"{row.get('json_file')}-{row.get('shape_idx')}-{row.get('label')}"


def advance_after_save(index: int, filtered_len: int, only_unreviewed: bool) -> int:
    if only_unreviewed:
        # Current row drops out of the filter; keep index so the next row slides into place.
        return min(index, max(filtered_len - 2, 0))
    return min(index + 1, max(filtered_len - 1, 0))


@st.cache_data(show_spinner=False)
def load_source_image_rgb(image_name: str, dataset_dir: str) -> np.ndarray | None:
    image_path = Path(dataset_dir) / image_name
    if not image_path.exists():
        return None
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return image_bgr[:, :, ::-1]


def draw_sku_bbox_on_context(
    context_rgb: np.ndarray,
    context_bbox: tuple[int, int, int, int],
    sku_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    cx1, cy1, _, _ = context_bbox
    sx1, sy1, sx2, sy2 = sku_bbox
    annotated = context_rgb.copy()
    cv2.rectangle(
        annotated,
        (sx1 - cx1, sy1 - cy1),
        (sx2 - cx1, sy2 - cy1),
        (255, 64, 64),
        2,
    )
    return annotated


def main() -> None:
    st.set_page_config(page_title="Golden SKU Review", layout="wide")
    st.title("Golden Dataset expected_sku Review")
    st.caption(
        "Priority task: fill expected_sku so true OCR accuracy can be measured. "
        "Use OCR suggestions as hints only. Use N/A when no SKU applies and explain why in notes."
    )

    truth_csv = Path(st.sidebar.text_input("Truth CSV", str(DEFAULT_TRUTH_CSV)))
    if not truth_csv.exists():
        st.error(f"Truth CSV not found: {truth_csv}")
        return

    truth_mtime = truth_csv.stat().st_mtime if truth_csv.exists() else 0.0
    if (
        "rows" not in st.session_state
        or st.session_state.get("truth_csv") != str(truth_csv)
        or st.session_state.get("truth_csv_mtime") != truth_mtime
    ):
        st.session_state.rows = load_rows(truth_csv)
        st.session_state.fieldnames = list(st.session_state.rows[0].keys()) if st.session_state.rows else []
        st.session_state.truth_csv = str(truth_csv)
        st.session_state.truth_csv_mtime = truth_mtime
        st.session_state.index = 0

    rows = st.session_state.rows
    fieldnames = st.session_state.fieldnames
    labels = sorted({row.get("label", "") for row in rows if row.get("label")})
    images = sorted({row.get("image", "") for row in rows if row.get("image")})
    batch_images = load_batch_images(DEFAULT_BATCH_IMAGES)
    batch_scope_options = ["Current review batch", "All images"]
    if not batch_images:
        batch_scope_options = ["All images"]

    label_filter = st.sidebar.selectbox("Label", ["All", *labels], index=0)
    review_scope = st.sidebar.selectbox("Review scope", batch_scope_options, index=0)
    scope_images = batch_images if review_scope == "Current review batch" else None
    scoped_images = sorted({row.get("image", "") for row in rows if row.get("image") and (not scope_images or row.get("image") in scope_images)})
    image_filter = st.sidebar.selectbox("Image", ["All", *scoped_images], index=0)
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    reviewer = st.sidebar.text_input("Reviewer name", value=st.session_state.get("reviewer", ""))
    st.session_state.reviewer = reviewer

    stretch_crops = st.sidebar.checkbox(
        "Stretch crop to column width",
        value=False,
        help="Off shows the crop at native pixel size (sharper). On enlarges to fill the column.",
    )
    reenhance_from_source = st.sidebar.checkbox(
        "Re-enhance from full-res source",
        value=True,
        help="Re-crops from the source image with stepwise upscaling + contrast sharpen. "
        "Better for tiny labels than older saved crop files.",
    )
    review_min_short = st.sidebar.slider(
        "Review upscale min short side",
        480,
        1080,
        720,
        step=80,
        help="Target short-side pixels when upscaling tiny label crops for review.",
    )
    show_enhanced = st.sidebar.checkbox(
        "Show enhanced contrast view",
        value=True,
        help="CLAHE + multi-scale sharpen. Off shows upscaled crop only.",
    )
    apply_deblur = st.sidebar.checkbox(
        "Apply deblur sharpening",
        value=True,
        help="Unsharp mask from production pipeline; can help faint digits.",
    )
    show_fullres_zoom = st.sidebar.checkbox(
        "Show full-res context zoom",
        value=False,
        help="Optional wide context view from the source image. Off by default — use label crop for review.",
    )
    context_pad_px = st.sidebar.slider("Context padding (px)", 40, 400, 160, step=20)
    context_scale = st.sidebar.slider("Context scale", 1.5, 6.0, 3.0, step=0.5)
    zoom_display_min_short = st.sidebar.slider(
        "Zoom display min short side",
        0,
        960,
        480,
        step=80,
        help="0 = native pixels only. Higher values upscale the zoom view for readability.",
    )

    filtered = filter_rows(rows, label_filter, image_filter, only_unreviewed, scope_images)
    scope_rows = filter_rows(rows, label_filter, "All", False, scope_images)
    total = len(scope_rows) if scope_rows else len(rows)
    reviewed = reviewed_count(scope_rows) if scope_rows else reviewed_count(rows)

    st.sidebar.metric("Reviewed", f"{reviewed} / {total}")
    st.sidebar.metric("Remaining", total - reviewed)
    st.sidebar.progress(reviewed / total if total else 0.0)
    if review_scope == "Current review batch" and batch_images:
        st.sidebar.caption(
            f"Batch: {len(batch_images)} images, {len(scope_rows)} SKU rows "
            f"({len(filtered)} in current filter)."
        )

    if not filtered:
        st.success("No rows match the current filters.")
        return

    if st.session_state.index >= len(filtered):
        st.session_state.index = 0

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("Previous", use_container_width=True):
            st.session_state.index = max(0, st.session_state.index - 1)
            st.rerun()
    with nav_col3:
        if st.button("Next", use_container_width=True):
            st.session_state.index = min(len(filtered) - 1, st.session_state.index + 1)
            st.rerun()

    row = filtered[st.session_state.index]
    current_key = row_key(row)
    with nav_col2:
        st.write(f"Row {st.session_state.index + 1} of {len(filtered)} in filter")

    left, right = st.columns([1, 1])

    with left:
        crop_path = resolve_project_data_path(row.get("crop_path", ""), PROJECT_ROOT)
        overlay_path = resolve_project_data_path(row.get("overlay_path", ""), PROJECT_ROOT)
        sku_bbox = (
            int(row.get("bbox_x1", 0)),
            int(row.get("bbox_y1", 0)),
            int(row.get("bbox_x2", 0)),
            int(row.get("bbox_y2", 0)),
        )
        bbox_w = sku_bbox[2] - sku_bbox[0]
        bbox_h = sku_bbox[3] - sku_bbox[1]

        if crop_path.exists():
            with Image.open(crop_path) as crop_img:
                crop_w, crop_h = crop_img.size
            st.caption(
                f"Label crop: {crop_w}×{crop_h}px saved file "
                f"(tight SKU bbox {bbox_w}×{bbox_h}px from annotation polygon)"
            )

            display_rgb: np.ndarray | None = None
            source_rgb = None
            ocr_hint = row.get("ocr_crop_suggestion", "")
            rotation_hint = row.get("rotation_ocr_suggestions", "")
            correction_angle = best_display_angle_from_rotation_hints(rotation_hint, ocr_hint)
            if reenhance_from_source:
                source_rgb = load_source_image_rgb(row.get("image", ""), str(DEFAULT_DATASET))
                if source_rgb is not None:
                    display_rgb = prepare_label_crop_for_review(
                        source_rgb,
                        sku_bbox,
                        min_short_side=review_min_short,
                        deblur=apply_deblur,
                        enhance=show_enhanced,
                        correction_angle=correction_angle,
                    )
                    dh, dw = display_rgb.shape[:2]
                    deskew_note = (
                        f", deskewed {correction_angle:g}° for readability"
                        if correction_angle
                        else ""
                    )
                    st.caption(
                        f"Review view: {dw}×{dh}px re-enhanced from source "
                        f"(native bbox {bbox_w}×{bbox_h}px{deskew_note})"
                    )

            if display_rgb is None:
                rgb = np.array(Image.open(crop_path).convert("RGB"))
                if show_enhanced:
                    display_rgb = enhance_crop_for_display(rgb, deblur=apply_deblur)
                else:
                    display_rgb = rgb

            st.image(
                display_rgb,
                caption=f"{row.get('label')} label crop",
                width="stretch" if stretch_crops else "content",
            )
        else:
            st.warning(f"Label crop not found: {crop_path}")

        if show_fullres_zoom:
            with st.expander("Full-res context zoom (optional)", expanded=True):
                if source_rgb is None:
                    source_rgb = load_source_image_rgb(row.get("image", ""), str(DEFAULT_DATASET))
                if source_rgb is not None:
                    zoom_rgb, context_bbox = crop_fullres_context(
                        source_rgb,
                        sku_bbox,
                        pad_px=context_pad_px,
                        scale=context_scale,
                        display_min_short_side=zoom_display_min_short,
                    )
                    zoom_rgb = draw_sku_bbox_on_context(zoom_rgb, context_bbox, sku_bbox)
                    if show_enhanced:
                        zoom_rgb = enhance_crop_for_display(zoom_rgb, deblur=apply_deblur)
                    zx1, zy1, zx2, zy2 = context_bbox
                    st.caption(
                        f"Context window {zx2 - zx1}×{zy2 - zy1}px — red box is the SKU label bbox, not a crop"
                    )
                    st.image(
                        zoom_rgb,
                        caption=f"{row.get('label')} — context zoom",
                        width="stretch",
                    )
                else:
                    st.warning(f"Source image not found in {DEFAULT_DATASET}")

        if overlay_path.exists():
            with st.expander("Full image overlay"):
                st.image(str(overlay_path), width="stretch")

    with right:
        st.subheader("Annotation")
        st.write(
            {
                "image": row.get("image"),
                "json_file": row.get("json_file"),
                "shape_idx": row.get("shape_idx"),
                "label": row.get("label"),
                "bbox": [
                    row.get("bbox_x1"),
                    row.get("bbox_y1"),
                    row.get("bbox_x2"),
                    row.get("bbox_y2"),
                ],
            }
        )

        ocr_hint = row.get("ocr_crop_suggestion", "")
        rotation_hint = row.get("rotation_ocr_suggestions", "")
        st.write("OCR suggestion", ocr_hint or "(none)")
        st.caption(
            "Auto deskew + multi-angle OCR runs in the background. "
            "Use this as a hint only — verify against the crop."
        )
        if rotation_hint:
            with st.expander("OCR debug (dev only)", expanded=False):
                st.code(rotation_hint, language=None)

        expected_sku = st.text_input(
            "expected_sku",
            value="",
            key=f"expected_sku_{current_key}",
            help=(
                "Enter exactly 6 or 10 characters: digits, plus X for any digit that "
                "is not visible. Use N/A when no SKU applies (notes required for N/A)."
            ),
        )

        notes = st.text_area(
            "notes",
            value=row.get("notes", ""),
            key=f"notes_{current_key}",
            help="Required when expected_sku is N/A (e.g. unreadable, no label visible, wrong region).",
        )
        action_col1, action_col2, action_col3 = st.columns(3)

        if action_col1.button("Save", use_container_width=True):
            stored_value, error = parse_expected_sku_input(expected_sku)
            if error:
                st.error(error)
            elif stored_value == "N/A" and not notes.strip():
                st.error("Notes are required when expected_sku is N/A.")
            else:
                row["expected_sku"] = stored_value
                row["review_status"] = "not_applicable" if stored_value == "N/A" else "reviewed"
                row["reviewer"] = reviewer
                row["notes"] = notes.strip()
                save_rows(truth_csv, rows, fieldnames)
                st.session_state.index = advance_after_save(
                    st.session_state.index,
                    len(filtered),
                    only_unreviewed,
                )
                st.rerun()

        if action_col2.button("Use OCR suggestion", use_container_width=True, disabled=not ocr_hint):
            row["expected_sku"] = parse_expected_sku_input(ocr_hint)[0] or ""
            row["review_status"] = "reviewed_ocr_assisted"
            row["reviewer"] = reviewer
            row["notes"] = notes.strip() or "Accepted OCR suggestion"
            save_rows(truth_csv, rows, fieldnames)
            st.session_state.index = advance_after_save(
                st.session_state.index,
                len(filtered),
                only_unreviewed,
            )
            st.rerun()

        if action_col3.button("Mark unreadable", use_container_width=True):
            row["expected_sku"] = "N/A"
            row["review_status"] = "not_applicable"
            row["reviewer"] = reviewer
            row["notes"] = notes.strip() or "Marked unreadable"
            save_rows(truth_csv, rows, fieldnames)
            st.session_state.index = advance_after_save(
                st.session_state.index,
                len(filtered),
                only_unreviewed,
            )
            st.rerun()

    st.divider()
    st.markdown(
        "**After review:** run `python scripts/golden_dataset/sync_expected_sku_from_truth_csv.py`, "
        "then `python scripts/golden_dataset/run_golden_dataset_local.py --mode ocr-crops` to get true accuracy."
    )


if __name__ == "__main__":
    main()

"""Technical model-review tab for OD, segmentation, and OCR comparisons."""

from __future__ import annotations

import csv
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from common.prediction_suggestions import (
    available_prediction_images,
    load_prediction_tracks_for_image,
    match_prediction_to_row,
    prediction_text,
)
from common.sku_review import is_na_expected_sku, normalize_sku_digits


def bbox_tuple(value: dict | None) -> tuple[int, int, int, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        bbox = tuple(int(value[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def row_bbox(row: dict | None) -> tuple[int, int, int, int] | None:
    if not row:
        return None
    return bbox_tuple(
        {
            "x1": row.get("bbox_x1"),
            "y1": row.get("bbox_y1"),
            "x2": row.get("bbox_x2"),
            "y2": row.get("bbox_y2"),
        }
    )


def segmentation_polygons(track: dict, block_name: str) -> list[list[list[int]]]:
    geometry = (
        track.get("segmentation", {})
        .get(block_name, {})
        .get("original_image", {})
    )
    polygons = geometry.get("polygons") or []
    if polygons:
        return polygons
    return [
        rect.get("box_points", [])
        for rect in geometry.get("rotated_rects") or []
        if rect.get("box_points")
    ]


def load_ground_truth_polygon(dataset_dir: Path, row: dict | None) -> list[list[float]]:
    if not row:
        return []
    json_path = dataset_dir / str(row.get("json_file", ""))
    if not json_path.exists():
        return []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        shape = payload.get("shapes", [])[int(row.get("shape_idx", -1))]
    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
        return []
    points = []
    for point in shape.get("points") or []:
        if isinstance(point, dict):
            x = point.get("x", point.get("0"))
            y = point.get("y", point.get("1"))
        else:
            x, y = point[:2]
        if x is not None and y is not None:
            points.append([float(x), float(y)])
    return points


def context_bbox(
    image_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int] | None],
    polygon_groups: list[list[list[list[int]]] | list[list[float]]],
) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for box in boxes:
        if box:
            xs.extend((box[0], box[2]))
            ys.extend((box[1], box[3]))
    for polygons in polygon_groups:
        for polygon in polygons:
            for point in polygon:
                if len(point) >= 2:
                    xs.append(int(point[0]))
                    ys.append(int(point[1]))
    height, width = image_rgb.shape[:2]
    if not xs or not ys:
        return 0, 0, width, height
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    padding = max(40, int(max(x2 - x1, y2 - y1) * 0.35))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def draw_review_panel(
    image_rgb: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    *,
    box: tuple[int, int, int, int] | None = None,
    polygons: list[list[list[int]]] | list[list[float]] | None = None,
    color: tuple[int, int, int] = (0, 200, 80),
) -> np.ndarray:
    x1, y1, x2, y2 = crop_bbox
    panel = image_rgb[y1:y2, x1:x2].copy()
    thickness = max(2, round(max(panel.shape[:2]) / 250))
    if box:
        cv2.rectangle(
            panel,
            (box[0] - x1, box[1] - y1),
            (box[2] - x1, box[3] - y1),
            color,
            thickness,
        )
    for polygon in polygons or []:
        points = np.array(
            [[int(point[0]) - x1, int(point[1]) - y1] for point in polygon],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(panel, [points], True, color, thickness)
    return panel


def review_panel_heading(column, title: str) -> None:
    """Reserve equal title height so every image in a comparison row aligns."""
    column.markdown(
        (
            '<div style="min-height:4.5rem;display:flex;align-items:flex-start;">'
            f'<h3 style="margin:0;padding:0;">{title}</h3></div>'
        ),
        unsafe_allow_html=True,
    )


def _load_truth_rows(truth_csv: Path) -> list[dict]:
    with truth_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_model_review_status(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    return {
        str(image)
        for image in payload.get("completed_images", [])
        if str(image).strip()
    }


def save_model_review_status(status_path: Path, completed_images: set[str]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = status_path.with_suffix(f"{status_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        payload = {
            "completed_images": sorted(completed_images),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary_path = status_path.with_suffix(
            f"{status_path.suffix}.{os.getpid()}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, status_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def render_model_review_tab(
    truth_csv: Path,
    dataset_dir: Path,
    predictions_dir: Path,
    status_path: Path | None = None,
) -> None:
    st.header("Model Review")
    st.caption(
        "Technical comparison of ground truth, object detection, segmentation, and OCR. "
        "The annotation team can stay in the Annotation Team tab."
    )
    if not truth_csv.exists():
        st.error(f"Truth CSV not found: {truth_csv}")
        return
    if not predictions_dir.exists():
        st.error(f"Prediction export not found: {predictions_dir}")
        return

    rows = _load_truth_rows(truth_csv)
    available_images = sorted(
        image
        for image in available_prediction_images(predictions_dir)
        if (dataset_dir / image).exists()
    )
    if not available_images:
        st.warning("No prediction JSON files match source images in the dataset.")
        return

    status_path = status_path or truth_csv.parent / "model_review_status.json"
    completed_images = load_model_review_status(status_path)
    completed_images &= set(available_images)
    image_index = min(
        int(st.session_state.get("model_review_image_index", 0)),
        len(available_images) - 1,
    )
    st.session_state.model_review_image_index = image_index

    image_nav = st.columns([1, 4, 1])
    if image_nav[0].button(
        "← Previous image",
        key="model_review_previous_image",
        disabled=image_index == 0,
        use_container_width=True,
    ):
        st.session_state.model_review_image_index = image_index - 1
        st.session_state.model_review_detection_index = 0
        st.rerun()
    image_name = available_images[image_index]
    image_state = "Complete" if image_name in completed_images else "Not complete"
    image_nav[1].markdown(
        f"<div style='text-align:center'><b>Image {image_index + 1} of "
        f"{len(available_images)}</b><br>{image_name}<br>{image_state}</div>",
        unsafe_allow_html=True,
    )
    if image_nav[2].button(
        "Next image →",
        key="model_review_next_image",
        disabled=image_index >= len(available_images) - 1,
        use_container_width=True,
    ):
        st.session_state.model_review_image_index = image_index + 1
        st.session_state.model_review_detection_index = 0
        st.rerun()

    progress_columns = st.columns([4, 1])
    progress_columns[0].progress(
        len(completed_images) / len(available_images),
        text=(
            f"Model review progress: {len(completed_images)} of "
            f"{len(available_images)} images complete"
        ),
    )
    if image_name in completed_images:
        if progress_columns[1].button("Mark not complete", use_container_width=True):
            completed_images.remove(image_name)
            save_model_review_status(status_path, completed_images)
            st.rerun()
    elif progress_columns[1].button(
        "Mark image complete",
        key="model_review_mark_complete",
        use_container_width=True,
    ):
        completed_images.add(image_name)
        save_model_review_status(status_path, completed_images)
        if image_index < len(available_images) - 1:
            st.session_state.model_review_image_index = image_index + 1
            st.session_state.model_review_detection_index = 0
        st.rerun()

    image_rows = [row for row in rows if row.get("image") == image_name]
    tracks = load_prediction_tracks_for_image(predictions_dir, image_name, rows)
    if not tracks:
        st.warning(f"No detection tracks found for {image_name}")
        return
    track_index = min(
        int(st.session_state.get("model_review_detection_index", 0)),
        len(tracks) - 1,
    )
    st.session_state.model_review_detection_index = track_index
    detection_nav = st.columns([1, 4, 1])
    if detection_nav[0].button(
        "← Previous detection",
        key="model_review_previous_detection",
        disabled=track_index == 0,
        use_container_width=True,
    ):
        st.session_state.model_review_detection_index = track_index - 1
        st.rerun()
    track = tracks[track_index]
    detection_nav[1].markdown(
        f"<div style='text-align:center'><b>Detection {track_index + 1} of "
        f"{len(tracks)}</b><br>det {track.get('det_id')} · "
        f"{track.get('class_name')} · confidence "
        f"{float(track.get('confidence', 0)):.3f} · OCR "
        f"{prediction_text(track) or 'none'}</div>",
        unsafe_allow_html=True,
    )
    if detection_nav[2].button(
        "Next detection →",
        key="model_review_next_detection",
        disabled=track_index >= len(tracks) - 1,
        use_container_width=True,
    ):
        st.session_state.model_review_detection_index = track_index + 1
        st.rerun()

    matched_row, overlap = match_prediction_to_row(track, image_rows)
    image_rgb = np.array(Image.open(dataset_dir / image_name).convert("RGB"))

    gt_bbox = row_bbox(matched_row)
    raw_bbox = bbox_tuple(track.get("orig_bbox"))
    buffered_bbox = bbox_tuple(track.get("buffered_bbox"))
    gt_polygon = load_ground_truth_polygon(dataset_dir, matched_row)
    raw_segmentation = segmentation_polygons(track, "raw_prediction")
    postprocessed_segmentation = segmentation_polygons(
        track,
        "postprocessed_minAreaRect",
    )
    review_context = context_bbox(
        image_rgb,
        [gt_bbox, raw_bbox, buffered_bbox],
        [[gt_polygon] if gt_polygon else [], raw_segmentation, postprocessed_segmentation],
    )

    top_columns = st.columns(3)
    top_panels = (
        ("Ground Truth OD", gt_bbox, (210, 35, 55)),
        ("Prediction Raw OD", raw_bbox, (30, 175, 85)),
        ("Prediction Buffered OD", buffered_bbox, (30, 175, 85)),
    )
    for column, (title, box, color) in zip(top_columns, top_panels):
        review_panel_heading(column, title)
        column.image(
            draw_review_panel(image_rgb, review_context, box=box, color=color),
            width="stretch",
        )
        if box is None:
            column.caption("Not available")

    bottom_columns = st.columns(3)
    bottom_panels = (
        (
            "Ground Truth SKU Region",
            [gt_polygon] if gt_polygon else [],
            (210, 35, 55),
        ),
        ("Raw Segmentation", raw_segmentation, (245, 145, 35)),
        ("Postprocessed Segmentation", postprocessed_segmentation, (20, 155, 210)),
    )
    for column, (title, polygons, color) in zip(bottom_columns, bottom_panels):
        review_panel_heading(column, title)
        column.image(
            draw_review_panel(
                image_rgb,
                review_context,
                polygons=polygons,
                color=color,
            ),
            width="stretch",
        )
        if not polygons:
            column.caption("Not available")

    expected_sku = str((matched_row or {}).get("expected_sku", "") or "").strip()
    predicted_sku = prediction_text(track)
    if not expected_sku:
        comparison = "Ground truth not reviewed"
    elif is_na_expected_sku(expected_sku):
        comparison = "Ground truth is non-scorable"
    elif normalize_sku_digits(expected_sku) == normalize_sku_digits(predicted_sku):
        comparison = "Correct"
    else:
        comparison = "Incorrect"

    sku_columns = st.columns(2)
    sku_columns[0].markdown("**Ground truth SKU**")
    sku_columns[0].code(expected_sku or "Unreviewed", language=None)
    sku_columns[1].markdown("**Predicted SKU**")
    sku_columns[1].code(predicted_sku or "None", language=None)

    metric_columns = st.columns(4)
    metric_columns[0].metric("OCR result", comparison)
    metric_columns[1].metric("OD IoU", f"{overlap:.3f}")
    metric_columns[2].metric(
        "OD confidence",
        f"{float(track.get('confidence', 0)):.3f}",
    )
    metric_columns[3].metric(
        "Segmentation",
        "Found" if track.get("seg_found") else "Not found",
    )

    with st.expander("Selected prediction details"):
        st.json(
            {
                "image": image_name,
                "det_id": track.get("det_id"),
                "class_name": track.get("class_name"),
                "matched_region_key": (matched_row or {}).get("region_key"),
                "ocr_words": track.get("ocr_words", []),
                "white_pixels": track.get("segmentation", {}).get("white_pixels"),
            }
        )

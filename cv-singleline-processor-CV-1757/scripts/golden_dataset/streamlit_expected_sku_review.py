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
import fcntl
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
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
from common.paths import resolve_project_data_path, setup_script_paths
from common.prediction_suggestions import prediction_suggestions_for_rows
from common.sku_review import (
    SCORABLE_REVIEW_STATUS,
    is_reviewed_annotation,
    is_scorable_review,
    normalize_sku_digits,
    parse_expected_sku_input,
    sku_digit_class,
)
from golden_dataset.model_review import render_model_review_tab

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)
DEFAULT_TRUTH_CSV = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "golden_sku_truth.csv"
DEFAULT_BATCH_IMAGES = (
    PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "review_batch_images.txt"
)
DEFAULT_REVIEW_ASSIGNMENTS = (
    PROJECT_ROOT
    / "research_outputs"
    / "golden_dataset_local_tests"
    / "reviewer_image_assignments.json"
)
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "predictions_json_export"
if not DEFAULT_PREDICTIONS_PATH.exists():
    DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "predictions.json"
DEFAULT_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"
DEFAULT_SCORABLE_TARGET = 350


def load_rows(truth_csv: Path) -> list[dict]:
    with truth_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prediction_path_signature(path: Path) -> tuple[int, int, int]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    stats = [item.stat() for item in paths if item.exists()]
    return (
        len(stats),
        max((stat.st_mtime_ns for stat in stats), default=0),
        sum(stat.st_size for stat in stats),
    )


@st.cache_data(show_spinner="Loading pipeline predictions...")
def load_cached_prediction_suggestions(
    predictions_path: str,
    signature: tuple[int, int, int],
    geometry_rows: list[dict],
) -> tuple[set[str], dict[str, dict]]:
    del signature
    return prediction_suggestions_for_rows(Path(predictions_path), geometry_rows)


def save_annotation(
    truth_csv: Path,
    key: str,
    updates: dict[str, str],
) -> tuple[list[dict], list[str]]:
    """Atomically merge one review so concurrent reviewers cannot overwrite each other."""
    lock_path = truth_csv.with_suffix(f"{truth_csv.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows = load_rows(truth_csv)
        fieldnames = list(rows[0].keys()) if rows else []
        for field in updates:
            if field not in fieldnames:
                fieldnames.append(field)

        matching_row = next((candidate for candidate in rows if row_key(candidate) == key), None)
        if matching_row is None:
            raise KeyError(f"Review row no longer exists: {key}")
        matching_row.update(updates)

        temporary_path = truth_csv.with_suffix(f"{truth_csv.suffix}.{os.getpid()}.tmp")
        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary_path, truth_csv)
        finally:
            temporary_path.unlink(missing_ok=True)
        return rows, fieldnames


def rebalance_reviewer_assignments(
    rows: list[dict],
    members: list[str],
    prior_assignments: dict[str, str] | None = None,
) -> dict[str, str]:
    """Balance unstarted whole images while pinning images that already have reviews."""
    if not members:
        return {}
    prior_assignments = prior_assignments or {}
    by_image: dict[str, list[dict]] = {}
    for row in rows:
        image = str(row.get("image", "") or "").strip()
        if image:
            by_image.setdefault(image, []).append(row)

    assignments: dict[str, str] = {}
    member_lookup = {member.casefold(): member for member in members}
    loads = {member: 0 for member in members}
    unstarted: list[tuple[str, int]] = []

    for image, image_rows in by_image.items():
        reviewed_rows = [
            row
            for row in image_rows
            if is_reviewed_annotation(
                row.get("review_status", ""),
                row.get("expected_sku", ""),
                row.get("notes", ""),
            )
        ]
        if not reviewed_rows:
            unstarted.append((image, len(image_rows)))
            continue

        reviewer_counts = Counter(
            str(row.get("reviewer", "") or "").strip()
            for row in reviewed_rows
            if str(row.get("reviewer", "") or "").strip()
        )
        recorded_owner = reviewer_counts.most_common(1)[0][0] if reviewer_counts else ""
        if not recorded_owner:
            recorded_owner = prior_assignments.get(image, "")
        owner = member_lookup.get(recorded_owner.casefold()) if recorded_owner else None
        if owner is None:
            # The prior annotator was removed. Preserve their saved rows, but make
            # the complete image available to a current reviewer for any remainder.
            unstarted.append((image, len(image_rows)))
            continue
        assignments[image] = owner
        loads[owner] += len(image_rows)

    for image, crop_count in sorted(unstarted, key=lambda item: (-item[1], item[0])):
        owner = min(members, key=lambda member: (loads[member], member.casefold()))
        assignments[image] = owner
        loads[owner] += crop_count
    return assignments


def register_reviewer_and_assign(
    assignment_path: Path,
    reviewer: str,
    rows: list[dict],
) -> tuple[list[str], dict[str, str]]:
    """Register one reviewer and atomically rebalance unstarted image groups."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer name is required")
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = assignment_path.with_suffix(f"{assignment_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state: dict = {}
        if assignment_path.exists():
            try:
                state = json.loads(assignment_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                state = {}

        members = [
            str(member).strip()
            for member in state.get("members", [])
            if str(member).strip()
        ]
        existing = next(
            (member for member in members if member.casefold() == reviewer.casefold()),
            None,
        )
        if existing is None:
            members.append(reviewer)
        else:
            reviewer = existing

        assignments = rebalance_reviewer_assignments(
            rows,
            members,
            state.get("assignments", {}),
        )
        payload = {
            "members": members,
            "assignments": assignments,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary_path = assignment_path.with_suffix(
            f"{assignment_path.suffix}.{os.getpid()}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, assignment_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return members, assignments


def delete_reviewer_and_reassign(
    assignment_path: Path,
    reviewer_to_delete: str,
    rows: list[dict],
) -> tuple[list[str], dict[str, str]]:
    """Remove one reviewer and redistribute their image groups among those remaining."""
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = assignment_path.with_suffix(f"{assignment_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not assignment_path.exists():
            return [], {}
        try:
            state = json.loads(assignment_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}

        deleted_key = reviewer_to_delete.strip().casefold()
        members = [
            str(member).strip()
            for member in state.get("members", [])
            if str(member).strip() and str(member).strip().casefold() != deleted_key
        ]
        assignments = rebalance_reviewer_assignments(
            rows,
            members,
            state.get("assignments", {}),
        )
        payload = {
            "members": members,
            "assignments": assignments,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary_path = assignment_path.with_suffix(
            f"{assignment_path.suffix}.{os.getpid()}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, assignment_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return members, assignments


def reviewed_count(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if is_reviewed_annotation(
            row.get("review_status", ""),
            row.get("expected_sku", ""),
            row.get("notes", ""),
        )
    )


def annotation_summary(
    rows: list[dict],
    prediction_images: set[str] | None = None,
    prediction_hints: dict[str, dict] | None = None,
) -> dict[str, float | int]:
    prediction_images = prediction_images or set()
    prediction_hints = prediction_hints or {}
    reviewed = reviewed_count(rows)
    scorable_rows = [
        row
        for row in rows
        if is_scorable_review(
            row.get("review_status", ""),
            row.get("expected_sku", ""),
            row.get("notes", ""),
        )
    ]
    correct = sum(
        1
        for row in scorable_rows
        if normalize_sku_digits(row.get("expected_sku", ""))
        == normalize_sku_digits(
            prediction_hints.get(row_key(row), {}).get("text", "")
            if row.get("image") in prediction_images
            else row.get("ocr_crop_suggestion", "")
        )
    )
    scorable = len(scorable_rows)
    return {
        "total": len(rows),
        "reviewed": reviewed,
        "scorable": scorable,
        "non_scorable": reviewed - scorable,
        "correct": correct,
        "accuracy": correct / scorable if scorable else 0.0,
    }


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
    annotation_filter: str,
    batch_images: set[str] | None = None,
) -> list[dict]:
    filtered = rows

    if batch_images is not None:
        filtered = [row for row in filtered if row.get("image") in batch_images]

    if label and label != "All":
        filtered = [row for row in filtered if row.get("label") == label]

    if image and image != "All":
        filtered = [row for row in filtered if row.get("image") == image]

    if annotation_filter != "All":
        def status_matches(row: dict) -> bool:
            reviewed = is_reviewed_annotation(
                row.get("review_status", ""),
                row.get("expected_sku", ""),
                row.get("notes", ""),
            )
            scorable = is_scorable_review(
                row.get("review_status", ""),
                row.get("expected_sku", ""),
                row.get("notes", ""),
            )
            if annotation_filter == "Unreviewed":
                return not reviewed
            if annotation_filter == "Scorable":
                return scorable
            if annotation_filter == "Non-scorable":
                return reviewed and not scorable
            return True

        filtered = [
            row for row in filtered if status_matches(row)
        ]

    return filtered


def row_key(row: dict) -> str:
    if row.get("region_key"):
        return row["region_key"]
    return f"{row.get('json_file')}-{row.get('shape_idx')}-{row.get('label')}"


def advance_after_save(index: int, filtered_len: int, row_leaves_filter: bool) -> int:
    if row_leaves_filter:
        # Current row drops out of the filter; keep index so the next row slides into place.
        return min(index, max(filtered_len - 2, 0))
    return min(index + 1, max(filtered_len - 1, 0))


def next_region_key_after_save(
    rows: list[dict],
    index: int,
    wrap_same_image: bool = False,
) -> str:
    """Stay in this image while eligible crops remain, then advance images."""
    if not rows or index < 0 or index >= len(rows):
        return ""
    current_image = rows[index].get("image")
    for candidate in rows[index + 1 :]:
        if candidate.get("image") == current_image:
            return row_key(candidate)
    if wrap_same_image:
        for candidate in rows[:index]:
            if candidate.get("image") == current_image:
                return row_key(candidate)
    if index + 1 < len(rows):
        return row_key(rows[index + 1])
    return ""


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


def render_annotation_team_tab() -> None:
    st.header("Golden Dataset expected_sku Review")
    st.caption(
        "Enter a verified 6- or 10-character SKU (digits, with X for unclear digits), "
        "or use Mark X when it is not visible. "
        "Only verified Scorable rows contribute to OCR accuracy."
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
        for required_field in (
            "review_status",
            "scorability",
            "sku_digit_class",
            "reviewer",
            "notes",
            "verified",
        ):
            if required_field not in st.session_state.fieldnames:
                st.session_state.fieldnames.append(required_field)
            for loaded_row in st.session_state.rows:
                loaded_row.setdefault(required_field, "")
        st.session_state.truth_csv = str(truth_csv)
        st.session_state.truth_csv_mtime = truth_mtime
        st.session_state.index = 0

    rows = st.session_state.rows
    try:
        geometry_fields = (
            "region_key",
            "image",
            "label",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        )
        geometry_rows = [
            {field: row.get(field, "") for field in geometry_fields}
            for row in rows
        ]
        prediction_images, prediction_hints = load_cached_prediction_suggestions(
            str(DEFAULT_PREDICTIONS_PATH),
            prediction_path_signature(DEFAULT_PREDICTIONS_PATH),
            geometry_rows,
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        prediction_images, prediction_hints = set(), {}
        st.warning(f"Could not load predictions.json: {exc}")

    if not st.session_state.get("reviewer"):
        with st.form("reviewer_join_form"):
            st.subheader("Start or resume your assigned images")
            reviewer_name = st.text_input(
                "Your name",
                help=(
                    "New names automatically receive a balanced share of unstarted images. "
                    "Returning reviewers receive their existing work."
                ),
            )
            join_submitted = st.form_submit_button("Start reviewing", type="primary")
        if not join_submitted:
            return
        reviewer_name = reviewer_name.strip()
        if not reviewer_name:
            st.error("Enter your name to start reviewing.")
            return
        st.session_state.reviewer = reviewer_name
        st.session_state.index = 0
        st.rerun()

    reviewer = str(st.session_state.reviewer)
    reviewer_col, change_col = st.columns([5, 1])
    reviewer_col.success(f"Reviewing as: {reviewer}")
    if change_col.button("Change reviewer", use_container_width=True):
        st.session_state.pop("reviewer", None)
        st.session_state.index = 0
        st.rerun()

    labels = sorted({row.get("label", "") for row in rows if row.get("label")})
    batch_images = load_batch_images(DEFAULT_BATCH_IMAGES)
    batch_scope_options = ["All images", "Current review batch"]
    if not batch_images:
        batch_scope_options = ["All images"]

    label_filter = st.sidebar.selectbox("Label", ["All", *labels], index=0)
    review_scope = st.sidebar.selectbox("Dataset scope", batch_scope_options, index=0)
    base_scope_images = (
        batch_images
        if review_scope == "Current review batch"
        else {row.get("image", "") for row in rows if row.get("image")}
    )
    assignment_rows = [row for row in rows if row.get("image") in base_scope_images]
    members, image_assignments = register_reviewer_and_assign(
        DEFAULT_REVIEW_ASSIGNMENTS,
        reviewer,
        assignment_rows,
    )
    canonical_reviewer = next(
        member for member in members if member.casefold() == reviewer.casefold()
    )
    if canonical_reviewer != reviewer:
        reviewer = canonical_reviewer
        st.session_state.reviewer = canonical_reviewer
    scope_images = {
        image
        for image, assigned_reviewer in image_assignments.items()
        if assigned_reviewer.casefold() == reviewer.casefold()
    }
    assigned_crop_count = sum(1 for row in assignment_rows if row.get("image") in scope_images)
    st.sidebar.caption(
        f"{reviewer}: {len(scope_images)} complete images, {assigned_crop_count} SKU crops. "
        f"{len(members)} reviewer{'s' if len(members) != 1 else ''} currently share this pool."
    )
    with st.sidebar.expander("Manage annotators", expanded=False):
        st.caption(
            "Removing an annotator keeps their saved labels and redistributes their "
            "assigned images among the remaining annotators."
        )
        reviewer_to_delete = st.selectbox(
            "Annotator to remove",
            members,
            key="reviewer_to_delete",
        )
        confirm_delete = st.checkbox(
            f"Confirm removal of {reviewer_to_delete}",
            key="confirm_reviewer_delete",
        )
        if st.button(
            "Delete annotator",
            disabled=not confirm_delete,
            use_container_width=True,
            type="secondary",
        ):
            delete_reviewer_and_reassign(
                DEFAULT_REVIEW_ASSIGNMENTS,
                reviewer_to_delete,
                assignment_rows,
            )
            if reviewer_to_delete.casefold() == reviewer.casefold():
                st.session_state.pop("reviewer", None)
            st.session_state.index = 0
            st.rerun()
    pending_annotation_filter = st.session_state.pop(
        "pending_annotation_filter",
        None,
    )
    if pending_annotation_filter:
        st.session_state.annotation_status = pending_annotation_filter
    annotation_filter = st.sidebar.selectbox(
        "Annotation status",
        ["Unreviewed", "All", "Scorable", "Non-scorable"],
        index=0,
        key="annotation_status",
    )

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

    filtered = filter_rows(rows, label_filter, "All", annotation_filter, scope_images)
    scope_rows = filter_rows(rows, label_filter, "All", "All", scope_images)
    pending_region_key = st.session_state.pop("pending_annotation_region_key", None)
    if pending_region_key:
        pending_index = next(
            (
                index
                for index, candidate in enumerate(filtered)
                if row_key(candidate) == pending_region_key
            ),
            None,
        )
        if pending_index is not None:
            st.session_state.index = pending_index
    summary_rows = scope_rows
    summary = annotation_summary(summary_rows, prediction_images, prediction_hints)
    scorable_target = st.sidebar.number_input(
        "Scorable target",
        min_value=1,
        max_value=max(len(summary_rows), DEFAULT_SCORABLE_TARGET),
        value=DEFAULT_SCORABLE_TARGET,
        step=25,
    )

    st.sidebar.metric("Reviewed", f"{summary['reviewed']} / {summary['total']}")
    st.sidebar.metric("Remaining", summary["total"] - summary["reviewed"])
    st.sidebar.progress(summary["reviewed"] / summary["total"] if summary["total"] else 0.0)
    st.sidebar.caption(
        f"Scorable labels: {summary['scorable']} / {scorable_target} "
        f"({min(summary['scorable'] / scorable_target, 1.0):.0%})"
    )
    st.sidebar.progress(min(summary["scorable"] / scorable_target, 1.0))
    if review_scope == "Current review batch" and batch_images:
        st.sidebar.caption(
            f"Batch: {len(batch_images)} images, {len(scope_rows)} SKU rows "
            f"({len(filtered)} in current filter)."
        )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Total regions", summary["total"])
    metric_cols[1].metric("Reviewed", summary["reviewed"])
    metric_cols[2].metric("Scorable", summary["scorable"])
    metric_cols[3].metric("Non-scorable", summary["non_scorable"])
    metric_cols[4].metric("OCR correct", summary["correct"])
    metric_cols[5].metric(
        "OCR accuracy",
        f"{summary['accuracy']:.1%}" if summary["scorable"] else "N/A",
    )

    if not filtered:
        st.success("No rows match the current filters.")
        return

    if st.session_state.index >= len(filtered):
        st.session_state.index = 0

    row = filtered[st.session_state.index]
    current_key = row_key(row)
    assigned_image_names = sorted(scope_images)
    current_image_number = assigned_image_names.index(row.get("image")) + 1
    navigable_image_names = list(dict.fromkeys(item.get("image") for item in scope_rows))
    navigable_image_index = navigable_image_names.index(row.get("image"))
    first_region_key_by_image = {
        image: row_key(
            next(item for item in scope_rows if item.get("image") == image)
        )
        for image in navigable_image_names
    }
    current_image_rows = [
        candidate for candidate in scope_rows
        if candidate.get("image") == row.get("image")
    ]
    filtered_crop_indices = [
        index for index, item in enumerate(filtered)
        if item.get("image") == row.get("image")
    ]
    current_crop_position = filtered_crop_indices.index(st.session_state.index)

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button(
            "← Previous image",
            key="annotation_previous_image",
            disabled=navigable_image_index == 0,
            use_container_width=True,
        ):
            previous_image = navigable_image_names[navigable_image_index - 1]
            st.session_state.pending_annotation_filter = "All"
            st.session_state.pending_annotation_region_key = first_region_key_by_image[
                previous_image
            ]
            st.rerun()
    with nav_col2:
        st.write(
            f"Image {current_image_number} of {len(assigned_image_names)} assigned to "
            f"{reviewer} · Crop {current_crop_position + 1} of "
            f"{len(filtered_crop_indices)} in the current filter "
            f"({len(current_image_rows)} total SKU crops)"
        )
        if annotation_filter == "Unreviewed":
            st.caption(
                "Resume behavior: reopening the tool starts at the first unreviewed crop "
                "assigned to this reviewer."
            )
    with nav_col3:
        if st.button(
            "Next image →",
            key="annotation_next_image",
            disabled=navigable_image_index >= len(navigable_image_names) - 1,
            use_container_width=True,
        ):
            next_image = navigable_image_names[navigable_image_index + 1]
            st.session_state.pending_annotation_filter = "All"
            st.session_state.pending_annotation_region_key = first_region_key_by_image[
                next_image
            ]
            st.rerun()

    crop_nav1, crop_nav2, crop_nav3 = st.columns([1, 2, 1])
    with crop_nav1:
        if st.button(
            "← Previous crop",
            key="annotation_previous_crop",
            disabled=current_crop_position == 0,
            use_container_width=True,
        ):
            st.session_state.index = filtered_crop_indices[current_crop_position - 1]
            st.rerun()
    crop_nav2.caption(
        f"Crop {current_crop_position + 1} of {len(filtered_crop_indices)} "
        f"for {row.get('image')}"
    )
    with crop_nav3:
        if st.button(
            "Next crop →",
            key="annotation_next_crop",
            disabled=current_crop_position >= len(filtered_crop_indices) - 1,
            use_container_width=True,
        ):
            st.session_state.index = filtered_crop_indices[current_crop_position + 1]
            st.rerun()

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
        form_section = st.container(border=True)
        prediction_hint = prediction_hints.get(current_key)
        uses_predictions_json = row.get("image") in prediction_images
        ocr_hint = (
            str(prediction_hint.get("text", ""))
            if prediction_hint
            else "" if uses_predictions_json
            else row.get("ocr_crop_suggestion", "")
        )
        form_section.subheader("OCR suggestion and expected SKU")
        form_section.info(ocr_hint or "No OCR suggestion available")
        if ocr_hint:
            form_section.caption(
                f"Suggestion class: {sku_digit_class(ocr_hint) or 'invalid length'}"
            )
        form_section.caption(
            (
                "Source: predictions.json pipeline output. "
                if uses_predictions_json
                else "Source: legacy precomputed crop OCR. "
            )
            + "Verify this hint against the crop before accepting it."
        )

        current_expected = str(row.get("expected_sku", "") or "").strip()
        if current_expected.upper() in {"N/A", "X"}:
            current_expected = ""
        expected_sku = form_section.text_input(
            "Verified expected SKU",
            value=current_expected,
            key=f"expected_sku_{current_key}",
            help=(
                "Enter exactly 6 or 10 characters: digits, plus X for any digit that "
                "is not visible. Use Mark X when the whole SKU is not visible."
            ),
        )
        entered_class = sku_digit_class(expected_sku)
        if expected_sku:
            form_section.caption(
                f"Classification: {entered_class or 'invalid — enter 6 or 10 chars (digits/X)'}"
            )

        notes = form_section.text_area(
            "notes",
            value=row.get("notes", ""),
            key=f"notes_{current_key}",
            help="Optional context explaining the review decision.",
        )
        action_col1, action_col2, action_col3 = form_section.columns(3)

        def persist(updates: dict[str, str]) -> bool:
            if not reviewer.strip():
                form_section.error("Enter your reviewer name before saving.")
                return False
            try:
                latest_rows, latest_fieldnames = save_annotation(truth_csv, current_key, updates)
            except (KeyError, OSError) as exc:
                form_section.error(f"Could not save this annotation: {exc}")
                return False
            st.session_state.rows = latest_rows
            st.session_state.fieldnames = latest_fieldnames
            st.session_state.truth_csv_mtime = truth_csv.stat().st_mtime
            return True

        def advance_to_next_crop() -> None:
            target_key = next_region_key_after_save(
                filtered,
                st.session_state.index,
                wrap_same_image=annotation_filter == "Unreviewed",
            )
            if target_key:
                st.session_state.pending_annotation_region_key = target_key
            else:
                st.session_state.index = advance_after_save(
                    st.session_state.index,
                    len(filtered),
                    annotation_filter == "Unreviewed",
                )
            st.rerun()

        if action_col1.button("Save", use_container_width=True):
            stored_value, error = parse_expected_sku_input(expected_sku)
            if error or stored_value in {"N/A", "X"}:
                form_section.error(
                    error or "A verified SKU must be exactly 6 or 10 characters (digits/X)."
                )
                stored_value = None

            if stored_value:
                updates = {
                    "expected_sku": stored_value,
                    "review_status": SCORABLE_REVIEW_STATUS,
                    "scorability": "scorable",
                    "sku_digit_class": sku_digit_class(stored_value),
                    "reviewer": reviewer.strip(),
                    "verified": "yes",
                    "notes": notes.strip(),
                }
                if persist(updates):
                    advance_to_next_crop()

        if action_col2.button("Use OCR suggestion", use_container_width=True, disabled=not ocr_hint):
            suggested_value, suggestion_error = parse_expected_sku_input(ocr_hint)
            if suggestion_error or suggested_value in {None, "N/A", "X"}:
                form_section.error(
                    suggestion_error
                    or "OCR suggestion is not a valid 6- or 10-character SKU (digits/X)."
                )
            else:
                updates = {
                    "expected_sku": suggested_value,
                    "review_status": SCORABLE_REVIEW_STATUS,
                    "scorability": "scorable",
                    "sku_digit_class": sku_digit_class(suggested_value),
                    "reviewer": reviewer.strip(),
                    "verified": "yes",
                    "notes": notes.strip() or "Accepted OCR suggestion",
                }
                if persist(updates):
                    advance_to_next_crop()

        if action_col3.button("Mark X (not visible)", use_container_width=True):
            updates = {
                "expected_sku": "X",
                "review_status": "non_scorable",
                "scorability": "non-scorable",
                "sku_digit_class": "not-visible",
                "reviewer": reviewer.strip(),
                "verified": "yes",
                "notes": notes.strip() or "SKU location not visible",
            }
            if persist(updates):
                advance_to_next_crop()

        metadata_section = st.container(border=True)
        metadata_section.subheader("Annotation metadata")
        metadata_section.write(
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

    st.divider()
    st.markdown(
        "**After review:** run `python scripts/golden_dataset/sync_expected_sku_from_truth_csv.py`, "
        "then `python scripts/golden_dataset/run_golden_dataset_local.py --mode ocr-crops` to get true accuracy."
    )


def main() -> None:
    st.set_page_config(page_title="Golden SKU Review", layout="wide")
    st.title("Golden SKU Annotation and Model Review")
    annotation_tab, model_review_tab = st.tabs(["Annotation Team", "Model Review"])
    with annotation_tab:
        render_annotation_team_tab()
    with model_review_tab:
        render_model_review_tab(
            DEFAULT_TRUTH_CSV,
            DEFAULT_DATASET,
            DEFAULT_PREDICTIONS_PATH,
        )


if __name__ == "__main__":
    main()

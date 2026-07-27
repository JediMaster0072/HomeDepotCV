"""
Streamlit app for auditing camera_cart EmptyItem predictions.

Usage:
  pip install streamlit pandas pillow
  cd cv-singleline-processor-CV-1757
  streamlit run scripts/empty_shelf/streamlit_empty_region_review.py

Generate the review CSV first (optional; app auto-creates it):
  python scripts/empty_shelf/generate_empty_region_truth_csv.py

After review, compute detector precision:
  python scripts/empty_shelf/evaluate_empty_shelf_detector_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)

import streamlit as st

from empty_shelf_review_utils import (
    DEFAULT_OVERLAY_DIR,
    DEFAULT_TEMPORAL_CSV,
    DEFAULT_TEMPORAL_DIR,
    DEFAULT_TRUTH_CSV,
    compute_detector_audit_metrics,
    ensure_truth_csv,
    normalize_label,
    save_truth_rows,
)


def filter_rows(rows: list[dict], image_idx: str, only_unreviewed: bool) -> list[dict]:
    filtered = rows

    if image_idx and image_idx != "All":
        filtered = [row for row in filtered if str(row.get("image_idx")) == image_idx]

    if only_unreviewed:
        filtered = [
            row
            for row in filtered
            if normalize_label(row.get("is_true_empty", "")) not in {"yes", "no", "unsure"}
        ]

    return filtered


def reviewed_count(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if normalize_label(row.get("is_true_empty", "")) in {"yes", "no", "unsure"}
    )


def save_label(row: dict, label: str, reviewer: str, notes: str, truth_csv: Path, rows: list[dict]) -> None:
    row["is_true_empty"] = label
    row["review_status"] = "reviewed" if label in {"yes", "no"} else "unsure"
    row["reviewer"] = reviewer
    row["notes"] = notes
    save_truth_rows(truth_csv, rows)


def main() -> None:
    st.set_page_config(page_title="Empty Shelf Review", layout="wide")
    st.title("camera_cart EmptyItem Review")
    st.caption(
        "Review each detector-emitted empty region. Bold boxes on the overlay correspond to "
        "region labels like 1.2 (image 1, region 2)."
    )

    truth_csv = Path(st.sidebar.text_input("Truth CSV", str(DEFAULT_TRUTH_CSV)))
    temporal_csv = Path(st.sidebar.text_input("Temporal CSV", str(DEFAULT_TEMPORAL_CSV)))
    temporal_dir = Path(st.sidebar.text_input("Temporal image dir", str(DEFAULT_TEMPORAL_DIR)))
    overlay_dir = Path(st.sidebar.text_input("Overlay dir", str(DEFAULT_OVERLAY_DIR)))

    rows = ensure_truth_csv(truth_csv, temporal_csv, temporal_dir, overlay_dir)
    if "index" not in st.session_state:
        st.session_state.index = 0

    image_options = sorted({str(row.get("image_idx")) for row in rows}, key=int)
    image_filter = st.sidebar.selectbox("Image", ["All", *image_options], index=0)
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    reviewer = st.sidebar.text_input("Reviewer name", value=st.session_state.get("reviewer", ""))
    st.session_state.reviewer = reviewer

    filtered = filter_rows(rows, image_filter, only_unreviewed)
    metrics = compute_detector_audit_metrics(rows)

    st.sidebar.metric("Reviewed", f"{reviewed_count(rows)} / {len(rows)}")
    st.sidebar.metric("Precision (reviewed yes/no)", metrics["precision"])
    st.sidebar.metric("False positives", metrics["false_positive_count"])
    st.sidebar.progress(reviewed_count(rows) / len(rows) if rows else 0.0)

    if not filtered:
        st.success("No rows match the current filters.")
        if metrics["reviewed_predictions"]:
            st.json(metrics)
        return

    if st.session_state.index >= len(filtered):
        st.session_state.index = 0

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("Previous", use_container_width=True):
            st.session_state.index = max(0, st.session_state.index - 1)
    with nav3:
        if st.button("Next", use_container_width=True):
            st.session_state.index = min(len(filtered) - 1, st.session_state.index + 1)

    row = filtered[st.session_state.index]
    with nav2:
        st.write(f"Row {st.session_state.index + 1} of {len(filtered)} in filter")

    left, right = st.columns([1.2, 0.8])

    with left:
        overlay_path = Path(row.get("overlay_path", ""))
        source_path = Path(row.get("source_image_path", ""))

        st.subheader(f"Review region {row.get('region_label')}")
        if overlay_path.exists():
            st.image(str(overlay_path), caption=f"Overlay for image {row.get('image_idx')}", use_container_width=True)
        else:
            st.warning(f"Overlay not found: {overlay_path}")

        if source_path.exists():
            with st.expander("Original shelf image"):
                st.image(str(source_path), use_container_width=True)

    with right:
        st.write(
            {
                "region_key": row.get("region_key"),
                "url_tail": row.get("url_tail"),
                "slot_id": row.get("slot_id"),
                "shelf_center": (row.get("cx"), row.get("cy")),
                "shelf_bbox": [
                    row.get("shelf_x1"),
                    row.get("shelf_y1"),
                    row.get("shelf_x2"),
                    row.get("shelf_y2"),
                ],
                "detector_source": row.get("detector_source"),
                "current_label": row.get("is_true_empty") or "(unreviewed)",
            }
        )
        st.info(
            "On the overlay, find the bold box/dot labeled "
            f"**{row.get('region_label')}**. Is that region truly empty?"
        )

        notes = st.text_area("notes", value=row.get("notes", ""))
        yes_col, no_col, unsure_col = st.columns(3)

        if yes_col.button("True empty", use_container_width=True):
            save_label(row, "yes", reviewer, notes, truth_csv, rows)
            st.success("Saved as true empty")
            st.rerun()

        if no_col.button("False positive", use_container_width=True):
            save_label(row, "no", reviewer, notes, truth_csv, rows)
            st.warning("Saved as false positive")
            st.rerun()

        if unsure_col.button("Unsure", use_container_width=True):
            save_label(row, "unsure", reviewer, notes, truth_csv, rows)
            st.info("Saved as unsure")
            st.rerun()

    st.divider()
    st.markdown(
        "**After review:** run `python scripts/empty_shelf/evaluate_empty_shelf_detector_audit.py` "
        "for precision / false-positive metrics."
    )
    with st.expander("Current audit metrics"):
        st.json(metrics)


if __name__ == "__main__":
    main()

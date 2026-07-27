"""Shared helpers for empty-shelf region review and camera_cart audit metrics."""

from __future__ import annotations

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from evaluate_empty_shelf_temporal_baseline import load_empty_regions
DEFAULT_TEMPORAL_CSV = PROJECT_ROOT / "temporal_data" / "Temporal_data_sample.csv"
DEFAULT_TEMPORAL_DIR = PROJECT_ROOT / "temporal_data"
DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "temporal_data_overlay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research_outputs" / "temporal_empty_analysis"
DEFAULT_TRUTH_CSV = DEFAULT_OUTPUT_DIR / "empty_region_truth.csv"

TRUTH_FIELDNAMES = [
    "region_key",
    "image_idx",
    "url_tail",
    "region_idx",
    "region_label",
    "captured_ts",
    "overlay_path",
    "source_image_path",
    "slot_id",
    "x_bin",
    "y_bin",
    "cx",
    "cy",
    "shelf_x1",
    "shelf_y1",
    "shelf_x2",
    "shelf_y2",
    "detector_source",
    "is_true_empty",
    "review_status",
    "reviewer",
    "notes",
]


def url_tail_to_filename_stem(url_tail: str) -> str:
    return url_tail.replace("#", "_")


def overlay_path_for(image_idx: int, url_tail: str, overlay_dir: Path) -> Path:
    stem = url_tail_to_filename_stem(url_tail)
    expected = overlay_dir / f"{image_idx:02d}_{stem}_empty_overlay.jpg"
    if expected.exists():
        return expected

    matches = sorted(overlay_dir.glob(f"{image_idx:02d}_*_empty_overlay.jpg"))
    if matches:
        return matches[0]

    return expected


def source_image_path(temporal_dir: Path, url_tail: str) -> Path:
    candidates = [
        temporal_dir / f"{url_tail_to_filename_stem(url_tail)}.jpg",
        temporal_dir / f"{url_tail}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    prefix = url_tail.split("#")[0]
    matches = sorted(temporal_dir.glob(f"{prefix}*.jpg"))
    if matches:
        return matches[0]

    return candidates[0]


def region_key(url_tail: str, region_idx: int) -> str:
    return f"{url_tail}::{region_idx}"


def build_truth_rows(
    temporal_csv: Path = DEFAULT_TEMPORAL_CSV,
    temporal_dir: Path = DEFAULT_TEMPORAL_DIR,
    overlay_dir: Path = DEFAULT_OVERLAY_DIR,
    bin_size: float = 0.5,
) -> list[dict]:
    regions, _ = load_empty_regions(temporal_csv, bin_size)
    rows: list[dict] = []

    for region in regions:
        image_idx = int(region["image_idx"])
        url_tail = region["url_tail"]
        region_idx = int(region["region_idx"])
        overlay_path = overlay_path_for(image_idx, url_tail, overlay_dir)
        source_path = source_image_path(temporal_dir, url_tail)

        rows.append(
            {
                "region_key": region_key(url_tail, region_idx),
                "image_idx": image_idx,
                "url_tail": url_tail,
                "region_idx": region_idx,
                "region_label": f"{image_idx}.{region_idx}",
                "captured_ts": region["captured_ts"],
                "overlay_path": str(overlay_path),
                "source_image_path": str(source_path),
                "slot_id": region["slot_id"],
                "x_bin": region["x_bin"],
                "y_bin": region["y_bin"],
                "cx": region["cx"],
                "cy": region["cy"],
                "shelf_x1": region["x1"],
                "shelf_y1": region["y1"],
                "shelf_x2": region["x2"],
                "shelf_y2": region["y2"],
                "detector_source": "camera_cart",
                "is_true_empty": "",
                "review_status": "needs_review",
                "reviewer": "",
                "notes": "",
            }
        )

    return rows


def load_truth_rows(truth_csv: Path) -> list[dict]:
    if not truth_csv.exists():
        return []

    with truth_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_truth_rows(truth_csv: Path, rows: list[dict]) -> None:
    truth_csv.parent.mkdir(parents=True, exist_ok=True)
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRUTH_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_truth_csv(
    truth_csv: Path = DEFAULT_TRUTH_CSV,
    temporal_csv: Path = DEFAULT_TEMPORAL_CSV,
    temporal_dir: Path = DEFAULT_TEMPORAL_DIR,
    overlay_dir: Path = DEFAULT_OVERLAY_DIR,
) -> list[dict]:
    existing = load_truth_rows(truth_csv)
    if existing:
        return existing

    rows = build_truth_rows(temporal_csv, temporal_dir, overlay_dir)
    save_truth_rows(truth_csv, rows)
    return rows


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z]", "", str(value or "").lower())


def reviewed_rows(rows: list[dict]) -> list[dict]:
    reviewed = []
    for row in rows:
        label = normalize_label(row.get("is_true_empty", ""))
        if label in {"yes", "no"}:
            reviewed.append(row)
    return reviewed


def compute_detector_audit_metrics(rows: list[dict]) -> dict:
    reviewed = reviewed_rows(rows)
    yes_rows = [row for row in reviewed if normalize_label(row.get("is_true_empty")) == "yes"]
    no_rows = [row for row in reviewed if normalize_label(row.get("is_true_empty")) == "no"]
    unsure_rows = [
        row for row in rows if normalize_label(row.get("is_true_empty", "")) in {"unsure", "unknown"}
    ]

    tp = len(yes_rows)
    fp = len(no_rows)
    reviewed_count = tp + fp
    precision = tp / reviewed_count if reviewed_count else 0.0

    by_image: dict[str, dict[str, int]] = {}
    for row in reviewed:
        key = row.get("region_label", "").split(".")[0]
        bucket = by_image.setdefault(key, {"yes": 0, "no": 0})
        if normalize_label(row.get("is_true_empty")) == "yes":
            bucket["yes"] += 1
        else:
            bucket["no"] += 1

    by_slot: dict[str, dict[str, int]] = {}
    for row in reviewed:
        slot = row.get("slot_id", "")
        bucket = by_slot.setdefault(slot, {"yes": 0, "no": 0})
        if normalize_label(row.get("is_true_empty")) == "yes":
            bucket["yes"] += 1
        else:
            bucket["no"] += 1

    return {
        "total_predictions": len(rows),
        "reviewed_predictions": reviewed_count,
        "true_empty_count": tp,
        "false_positive_count": fp,
        "unsure_count": len(unsure_rows),
        "remaining_unreviewed": len(rows) - reviewed_count - len(unsure_rows),
        "precision": round(precision, 4),
        "false_positive_rate": round(fp / reviewed_count, 4) if reviewed_count else 0.0,
        "recall": None,
        "f1": None,
        "recall_note": (
            "Recall is not measurable from camera_cart positives alone. "
            "Review only covers detector-emitted EmptyItem boxes."
        ),
        "by_image": by_image,
        "by_slot": by_slot,
    }

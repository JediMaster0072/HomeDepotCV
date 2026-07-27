"""
Compute audit metrics for the existing camera_cart EmptyItem detector.

Each reviewed row is a detector-emitted empty region. Human labels:
  yes   -> true empty (true positive)
  no    -> false positive
  unsure -> excluded from precision

Recall/F1 are not reported because this CSV only contains detector positives.

Usage:
  python scripts/empty_shelf/evaluate_empty_shelf_detector_audit.py
  python scripts/empty_shelf/evaluate_empty_shelf_detector_audit.py --truth-csv path/to/empty_region_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)

from empty_shelf_review_utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRUTH_CSV,
    compute_detector_audit_metrics,
    ensure_truth_csv,
    normalize_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit camera_cart EmptyItem detector precision.")
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = ensure_truth_csv(args.truth_csv)
    metrics = compute_detector_audit_metrics(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "empty_region_detector_audit_metrics.json"
    reviewed_path = args.output_dir / "empty_region_detector_audit_reviewed.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    reviewed_rows = [
        row
        for row in rows
        if normalize_label(row.get("is_true_empty", "")) in {"yes", "no", "unsure"}
    ]
    if reviewed_rows:
        with reviewed_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(reviewed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(reviewed_rows)

    print(f"Total camera_cart empty regions: {metrics['total_predictions']}")
    print(f"Reviewed: {metrics['reviewed_predictions']}  unsure: {metrics['unsure_count']}  remaining: {metrics['remaining_unreviewed']}")
    print(f"True empty: {metrics['true_empty_count']}")
    print(f"False positives: {metrics['false_positive_count']}")
    print(f"Precision (yes / (yes + no)): {metrics['precision']}")
    print(f"False positive rate: {metrics['false_positive_rate']}")
    print(metrics["recall_note"])
    print(f"Wrote {metrics_path}")
    if reviewed_rows:
        print(f"Wrote {reviewed_path}")


if __name__ == "__main__":
    main()

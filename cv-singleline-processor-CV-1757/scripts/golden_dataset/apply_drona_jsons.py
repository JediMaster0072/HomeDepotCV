"""
Apply updated Drona JSON annotations to the expected-SKU golden dataset copy.

Copies JSON from drona_jsons/ into Golden_Dataset_overhead_eval_expected_sku/,
adds expected_sku placeholder fields on every *_SKU shape, and optionally
preserves human labels from an existing golden_sku_truth.csv via bbox IoU match.

Usage:
  python scripts/golden_dataset/apply_drona_jsons.py --dry-run
  python scripts/golden_dataset/apply_drona_jsons.py
  python scripts/golden_dataset/apply_drona_jsons.py --drona-dir ../drona_jsons
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.golden_shapes import SKU_LABEL_SUFFIX, bbox_from_points, bbox_iou, shape_points
from common.paths import setup_script_paths

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)

DEFAULT_DRONA_DIR = PROJECT_ROOT / "drona_jsons"
DEFAULT_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"
DEFAULT_TRUTH_CSV = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "golden_sku_truth.csv"
DEFAULT_IOU = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Drona JSON annotations to expected-SKU dataset.")
    parser.add_argument("--drona-dir", type=Path, default=DEFAULT_DRONA_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def load_truth_labels(truth_csv: Path) -> list[dict]:
    if not truth_csv.exists():
        return []

    with truth_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truth_label_for_shape(image: str, label: str, bbox: tuple[int, int, int, int], truth_rows: list[dict], iou_threshold: float) -> dict | None:
    best_row = None
    best_iou = 0.0

    for row in truth_rows:
        if row.get("image") != image or row.get("label") != label:
            continue
        if not row.get("expected_sku", "").strip():
            continue

        try:
            old_bbox = (
                int(row["bbox_x1"]),
                int(row["bbox_y1"]),
                int(row["bbox_x2"]),
                int(row["bbox_y2"]),
            )
        except (KeyError, ValueError):
            continue

        score = bbox_iou(bbox, old_bbox)
        if score > best_iou:
            best_iou = score
            best_row = row

    if best_row is None or best_iou < iou_threshold:
        return None

    return {
        "expected_sku": best_row.get("expected_sku", "").strip(),
        "expected_sku_review_status": best_row.get("review_status", "").strip() or "reviewed",
        "expected_sku_source": "golden_sku_truth.csv",
        "expected_sku_reviewer": best_row.get("reviewer", "").strip(),
        "expected_sku_notes": best_row.get("notes", "").strip(),
        "match_iou": round(best_iou, 4),
    }


def apply_expected_sku_fields(shape: dict, preserved: dict | None) -> None:
    if preserved:
        shape["expected_sku"] = preserved["expected_sku"]
        shape["expected_sku_review_status"] = preserved["expected_sku_review_status"]
        shape["expected_sku_source"] = preserved["expected_sku_source"]
        if preserved.get("expected_sku_reviewer"):
            shape["expected_sku_reviewer"] = preserved["expected_sku_reviewer"]
        if preserved.get("expected_sku_notes"):
            shape["expected_sku_notes"] = preserved["expected_sku_notes"]
    else:
        shape["expected_sku"] = ""
        shape["expected_sku_review_status"] = "needs_review"
        shape["expected_sku_source"] = "drona_jsons"


def main() -> None:
    args = parse_args()

    if not args.drona_dir.is_dir():
        raise FileNotFoundError(f"Drona JSON dir not found: {args.drona_dir}")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset}")

    drona_files = sorted(args.drona_dir.glob("*.json"))
    if not drona_files:
        raise FileNotFoundError(f"No JSON files in {args.drona_dir}")

    truth_rows = load_truth_labels(args.truth_csv)

    if not args.no_backup and not args.dry_run:
        backup_root = args.dataset.parent / f"{args.dataset.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(args.dataset, backup_root, dirs_exist_ok=False)
        print(f"Backed up dataset to {backup_root}")

    updated_files = 0
    sku_shapes = 0
    preserved_labels = 0

    for drona_path in drona_files:
        data = json.loads(drona_path.read_text(encoding="utf-8"))
        image_name = drona_path.with_suffix(".jpg").name
        image_width = int(data.get("imageWidth") or 0)
        image_height = int(data.get("imageHeight") or 0)

        for shape_idx, shape in enumerate(data.get("shapes", [])):
            label = shape.get("label", "")
            if not str(label).endswith(SKU_LABEL_SUFFIX):
                continue

            points = shape_points(shape)
            bbox = bbox_from_points(points, image_width, image_height)
            if bbox is None:
                continue

            sku_shapes += 1
            preserved = truth_label_for_shape(image_name, label, bbox, truth_rows, args.iou_threshold)
            if preserved:
                preserved_labels += 1
            apply_expected_sku_fields(shape, preserved)

        out_path = args.dataset / drona_path.name
        updated_files += 1

        if not args.dry_run:
            out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    mode = "Would update" if args.dry_run else "Updated"
    print(f"{mode} {updated_files} JSON files from {args.drona_dir}")
    print(f"Total *_SKU shapes: {sku_shapes}")
    print(f"Preserved human labels from {args.truth_csv.name}: {preserved_labels}")
    if args.dry_run:
        print("Re-run without --dry-run to write files.")


if __name__ == "__main__":
    main()

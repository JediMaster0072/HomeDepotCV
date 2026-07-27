"""
Build / refresh golden_sku_truth.csv from the expected-SKU dataset JSONs.

One row per *_SKU polygon. Preserves existing reviewer fields when region_key
or bbox IoU matches a prior truth CSV row.

Usage:
  python scripts/golden_dataset/generate_golden_sku_truth_csv.py
  python scripts/golden_dataset/generate_golden_sku_truth_csv.py --run-crops
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

from common.crop_preprocess import DEFAULT_MIN_CROP_SHORT_SIDE
from common.golden_shapes import (
    SKU_LABEL_SUFFIX,
    bbox_from_points,
    bbox_iou,
    crop_filename,
    region_key,
    shape_points,
)
from common.paths import path_for_csv, setup_script_paths
from common.sku_review import is_reviewed_expected_sku

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.google_ocr_utils import primary_sku_suggestion

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)
DEFAULT_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests"
DEFAULT_TRUTH_CSV = DEFAULT_OUTPUT_DIR / "golden_sku_truth.csv"
DEFAULT_OVERLAY_DIR = DEFAULT_OUTPUT_DIR / "label_overlays_expected_sku"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_IOU = 0.65

FIELDNAMES = [
    "region_key",
    "image",
    "json_file",
    "shape_idx",
    "label",
    "expected_sku",
    "review_status",
    "reviewer",
    "notes",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_path",
    "overlay_path",
    "crop_exists",
    "ocr_crop_suggestion",
    "rotation_ocr_suggestions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden_sku_truth.csv from dataset JSONs.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--prior-truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU)
    parser.add_argument("--run-crops", action="store_true", help="Run crop generation before building CSV.")
    parser.add_argument("--force", action="store_true", help="Overwrite output CSV without backup.")
    return parser.parse_args()


def iter_image_json_pairs(dataset_dir: Path):
    for image_path in sorted(dataset_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            yield image_path, json_path


def load_prior_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match_prior_row(prior_rows: list[dict], image: str, label: str, bbox: tuple[int, int, int, int], key: str, iou_threshold: float) -> dict | None:
    for row in prior_rows:
        if row.get("region_key") == key:
            return row

    best = None
    best_iou = 0.0
    for row in prior_rows:
        if row.get("image") != image or row.get("label") != label:
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
            best = row

    if best is not None and best_iou >= iou_threshold:
        return best
    return None


def normalize_sku_text(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def ocr_suggestion_from_texts(ocr_texts: str) -> str:
    for part in str(ocr_texts or "").split("|"):
        digits = normalize_sku_text(part)
        if digits:
            return digits
    return ""


def load_ocr_crop_rows(path: Path) -> dict[tuple[str, int, str], dict]:
    if not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    indexed: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        try:
            key = (row["image"], int(row["shape_idx"]), row["label"])
        except (KeyError, ValueError):
            continue
        indexed[key] = row
    return indexed


def load_rotation_ocr_groups(path: Path) -> dict[tuple[str, int, str], list[dict]]:
    if not path.exists():
        return {}

    groups: dict[tuple[str, int, str], list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["image"], int(row["shape_idx"]), row["label"])
            except (KeyError, ValueError):
                continue
            groups.setdefault(key, []).append(row)
    return groups


def format_rotation_suggestions(group_rows: list[dict]) -> str:
    parts = []
    for row in sorted(group_rows, key=lambda item: float(item.get("rotation_angle", 0) or 0)):
        sku = ocr_suggestion_from_texts(row.get("ocr_texts", ""))
        if sku:
            parts.append(f"{row.get('rotation_angle', '')}:{sku}")
    return "|".join(parts)


def best_sku_from_rotation_group(group_rows: list[dict]) -> str:
    candidates: list[dict] = []
    for row in group_rows:
        sku = ocr_suggestion_from_texts(row.get("ocr_texts", ""))
        if not sku:
            continue
        candidates.append(
            {
                "text": sku,
                "source": f"rot{row.get('rotation_angle', '')}",
            }
        )
    if not candidates:
        return ""
    return primary_sku_suggestion(candidates)


def pick_best_ocr_hint(*hints: str) -> str:
    candidates = [{"text": hint, "source": ""} for hint in hints if hint]
    if not candidates:
        return ""
    return primary_sku_suggestion(candidates)


def build_rows(args: argparse.Namespace, prior_rows: list[dict]) -> list[dict]:
    rows = []
    crops_dir = args.output_dir / "crops"
    ocr_rows = load_ocr_crop_rows(args.output_dir / "ocr-crops_summary.csv")
    rotation_groups = load_rotation_ocr_groups(args.output_dir / "ocr-rotation-crops_summary.csv")

    for image_path, json_path in iter_image_json_pairs(args.dataset):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_width = int(data.get("imageWidth") or 0)
        image_height = int(data.get("imageHeight") or 0)
        overlay_path = args.overlay_dir / f"{image_path.stem}_labels.jpg"

        for shape_idx, shape in enumerate(data.get("shapes", [])):
            label = shape.get("label", "")
            if not str(label).endswith(SKU_LABEL_SUFFIX):
                continue

            points = shape_points(shape)
            bbox = bbox_from_points(points, image_width, image_height)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            key = region_key(image_path.name, label, bbox)
            crop_path = crops_dir / image_path.stem / crop_filename(shape_idx, label, bbox)
            prior = match_prior_row(prior_rows, image_path.name, label, bbox, key, args.iou_threshold)

            expected_sku = str(shape.get("expected_sku", "") or "").strip()
            review_status = str(shape.get("expected_sku_review_status", "") or "").strip()
            reviewer = str(shape.get("expected_sku_reviewer", "") or "").strip()
            notes = str(shape.get("expected_sku_notes", "") or "").strip()

            if prior:
                if is_reviewed_expected_sku(prior.get("expected_sku", "")):
                    expected_sku = prior.get("expected_sku", "").strip() or expected_sku
                    review_status = prior.get("review_status", "").strip() or review_status
                    reviewer = prior.get("reviewer", "").strip() or reviewer
                    notes = prior.get("notes", "").strip() or notes
                ocr_hint = prior.get("ocr_crop_suggestion", "").strip()
                rot_hint = prior.get("rotation_ocr_suggestions", "").strip()
            else:
                ocr_hint = ""
                rot_hint = ""

            ocr_row = ocr_rows.get((image_path.name, shape_idx, label))
            if ocr_row:
                fresh_hint = ocr_suggestion_from_texts(ocr_row.get("ocr_texts", ""))
                if fresh_hint:
                    ocr_hint = fresh_hint

            rot_group = rotation_groups.get((image_path.name, shape_idx, label), [])
            if rot_group:
                rot_hint = format_rotation_suggestions(rot_group) or rot_hint
                rot_best = best_sku_from_rotation_group(rot_group)
                if rot_best:
                    ocr_hint = pick_best_ocr_hint(ocr_hint, rot_best)

            rows.append(
                {
                    "region_key": key,
                    "image": image_path.name,
                    "json_file": json_path.name,
                    "shape_idx": shape_idx,
                    "label": label,
                    "expected_sku": expected_sku,
                    "review_status": review_status or ("reviewed" if expected_sku else "needs_review"),
                    "reviewer": reviewer,
                    "notes": notes,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "crop_path": path_for_csv(crop_path, PROJECT_ROOT),
                    "overlay_path": path_for_csv(overlay_path, PROJECT_ROOT),
                    "crop_exists": "yes" if crop_path.exists() else "no",
                    "ocr_crop_suggestion": ocr_hint,
                    "rotation_ocr_suggestions": rot_hint,
                }
            )

    return rows


def main() -> None:
    args = parse_args()

    if args.run_crops:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_golden_dataset_local.py"),
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(args.output_dir),
                "--mode",
                "crops",
                "--save-crops",
                "--min-crop-short-side",
                str(DEFAULT_MIN_CROP_SHORT_SIDE),
            ],
            check=True,
        )

    prior_rows = load_prior_rows(args.prior_truth_csv)
    if args.output_csv.exists() and not args.force:
        backup = args.output_csv.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy2(args.output_csv, backup)
        print(f"Backed up existing CSV to {backup}")

    rows = build_rows(args, prior_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    reviewed = sum(1 for row in rows if is_reviewed_expected_sku(row.get("expected_sku", "")))
    crops_exist = sum(1 for row in rows if row.get("crop_exists") == "yes")
    print(f"Wrote {args.output_csv}")
    print(f"Rows: {len(rows)}  reviewed: {reviewed}  crops on disk: {crops_exist}")


if __name__ == "__main__":
    main()

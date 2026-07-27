"""
Parse EmptyItem regions from temporal_data and compute first empty-shelf metrics.

This is the minimum reproducible baseline for priority task #2. Full pixel-level
precision/recall/F1 still requires spatial product labels; this script reports:
  - per-image empty-region count accuracy for a persistence baseline
  - slot-level precision/recall/F1 in shelf-coordinate bins
  - temporal consistency of recurring empty slots

Usage:
  python scripts/empty_shelf/evaluate_empty_shelf_temporal_baseline.py
  python scripts/empty_shelf/evaluate_empty_shelf_temporal_baseline.py --min-slot-images 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)
DEFAULT_TEMPORAL_CSV = PROJECT_ROOT / "temporal_data" / "Temporal_data_sample.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research_outputs" / "temporal_empty_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal empty-shelf baseline metrics.")
    parser.add_argument("--temporal-csv", type=Path, default=DEFAULT_TEMPORAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-size", type=float, default=0.5, help="Shelf-coordinate bin width.")
    parser.add_argument(
        "--min-slot-images",
        type=int,
        default=3,
        help="Minimum images a slot must appear empty to be treated as chronically empty.",
    )
    parser.add_argument(
        "--persistence-threshold",
        type=float,
        default=0.34,
        help="Prior-image fraction required to predict a shelf slot as empty.",
    )
    return parser.parse_args()


def parse_wkt_boxes(coordinates: str) -> list[tuple[float, float, float, float]]:
    if not coordinates:
        return []

    boxes = []
    point_pattern = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")
    for linestring in re.findall(r"\(([^()]+)\)", coordinates):
        points = [(float(x), float(y)) for x, y in point_pattern.findall(linestring)]
        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))

    return boxes


def bin_value(value: float, bin_size: float) -> float:
    return round(value / bin_size) * bin_size


def region_row(image_idx: int, captured_ts: str, url_tail: str, region_idx: int, box: tuple[float, float, float, float], bin_size: float) -> dict:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "image_idx": image_idx,
        "captured_ts": captured_ts,
        "url_tail": url_tail,
        "region_idx": region_idx,
        "x1": round(x1, 3),
        "y1": round(y1, 3),
        "x2": round(x2, 3),
        "y2": round(y2, 3),
        "cx": round(cx, 3),
        "cy": round(cy, 3),
        "w": round(x2 - x1, 3),
        "h": round(y2 - y1, 3),
        "x_bin": bin_value(cx, bin_size),
        "y_bin": bin_value(cy, bin_size),
        "slot_id": f"{bin_value(cx, bin_size):.1f},{bin_value(cy, bin_size):.1f}",
    }


def load_empty_regions(temporal_csv: Path, bin_size: float) -> tuple[list[dict], list[str]]:
    with temporal_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    image_order: list[str] = []
    seen = set()
    regions: list[dict] = []

    for row in rows:
        if row.get("classification") != "EmptyItem":
            continue

        url_tail = row.get("url", "").split("/")[-1]
        if url_tail not in seen:
            seen.add(url_tail)
            image_order.append(url_tail)

        image_idx = image_order.index(url_tail) + 1
        boxes = parse_wkt_boxes(row.get("coordinates", ""))

        for region_idx, box in enumerate(boxes, start=1):
            regions.append(
                region_row(
                    image_idx=image_idx,
                    captured_ts=row.get("captured_ts", ""),
                    url_tail=url_tail,
                    region_idx=region_idx,
                    box=box,
                    bin_size=bin_size,
                )
            )

    return regions, image_order


def compute_slot_metrics(
    regions: list[dict],
    image_order: list[str],
    min_slot_images: int,
    persistence_threshold: float,
) -> dict:
    gt_by_image: dict[str, set[str]] = defaultdict(set)
    for region in regions:
        gt_by_image[region["url_tail"]].add(region["slot_id"])

    slot_images: dict[str, set[str]] = defaultdict(set)
    for region in regions:
        slot_images[region["slot_id"]].add(region["url_tail"])

    tp = fp = fn = 0
    per_image_rows = []

    for idx, url_tail in enumerate(image_order):
        gt_slots = gt_by_image.get(url_tail, set())
        prior_images = set(image_order[:idx])

        pred_slots = set()
        for slot, images in slot_images.items():
            prior_hits = len(images & prior_images)
            if prior_images and (prior_hits / len(prior_images)) >= persistence_threshold:
                pred_slots.add(slot)
            elif not prior_images and len(images) >= min_slot_images:
                # First image fallback: use globally recurring slots.
                if (len(images) / len(image_order)) >= persistence_threshold:
                    pred_slots.add(slot)

        tp += len(gt_slots & pred_slots)
        fp += len(pred_slots - gt_slots)
        fn += len(gt_slots - pred_slots)

        gt_count = sum(1 for region in regions if region["url_tail"] == url_tail)
        pred_count = len(pred_slots)
        per_image_rows.append(
            {
                "url_tail": url_tail,
                "gt_empty_region_count": gt_count,
                "gt_empty_slot_count": len(gt_slots),
                "pred_empty_slot_count": pred_count,
                "count_abs_error": abs(gt_count - pred_count),
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    count_mae = sum(row["count_abs_error"] for row in per_image_rows) / len(per_image_rows)

    chronic_slots = {
        slot
        for slot, images in slot_images.items()
        if len(images) >= min_slot_images
    }

    return {
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slot_f1": round(f1, 4),
        "empty_count_mae": round(count_mae, 4),
        "chronic_slot_count": len(chronic_slots),
        "per_image": per_image_rows,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.temporal_csv.exists():
        raise FileNotFoundError(f"Temporal CSV not found: {args.temporal_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regions, image_order = load_empty_regions(args.temporal_csv, args.bin_size)
    metrics = compute_slot_metrics(
        regions,
        image_order,
        min_slot_images=args.min_slot_images,
        persistence_threshold=args.persistence_threshold,
    )

    summary_path = args.output_dir / "temporal_empty_region_summary.csv"
    metrics_path = args.output_dir / "temporal_empty_baseline_metrics.json"
    per_image_path = args.output_dir / "temporal_empty_baseline_per_image.csv"

    write_csv(summary_path, regions)
    write_csv(per_image_path, metrics.pop("per_image"))
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Parsed empty regions: {len(regions)} across {len(image_order)} images")
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_image_path}")
    print(f"Wrote {metrics_path}")
    print("Baseline slot metrics (persistence model):")
    print(f"  precision={metrics['slot_precision']}")
    print(f"  recall={metrics['slot_recall']}")
    print(f"  f1={metrics['slot_f1']}")
    print(f"  empty_count_mae={metrics['empty_count_mae']}")
    print(f"  chronic_slots={metrics['chronic_slot_count']}")


if __name__ == "__main__":
    main()

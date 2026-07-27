"""Generate repeatable OD, crop, segmentation, and OCR evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths
from common.prediction_suggestions import (
    available_prediction_images,
    load_prediction_tracks_for_image,
    prediction_text,
)
from common.sku_review import (
    is_scorable_review,
    normalize_sku_digits,
    sku_digit_class,
)

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)
DEFAULT_TRUTH_CSV = (
    PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "golden_sku_truth.csv"
)
DEFAULT_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "predictions_json_export"
DEFAULT_OUTPUT = PROJECT_ROOT / "research_outputs" / "overhead_pipeline_evaluation"


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate overhead OD, crop coverage, segmentation, and OCR.",
    )
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-failure-overlays", type=int, default=20)
    return parser.parse_args()


def box_from_dict(value: dict | None) -> Box | None:
    if not isinstance(value, dict):
        return None
    try:
        box = Box(*(int(value[key]) for key in ("x1", "y1", "x2", "y2")))
    except (KeyError, TypeError, ValueError):
        return None
    return box if box.area > 0 else None


def box_from_row(row: dict) -> Box | None:
    return box_from_dict(
        {
            "x1": row.get("bbox_x1"),
            "y1": row.get("bbox_y1"),
            "x2": row.get("bbox_x2"),
            "y2": row.get("bbox_y2"),
        }
    )


def intersection_area(a: Box, b: Box) -> int:
    return max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0,
        min(a.y2, b.y2) - max(a.y1, b.y1),
    )


def box_iou(a: Box, b: Box) -> float:
    intersection = intersection_area(a, b)
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


def ground_truth_coverage(prediction: Box, ground_truth: Box) -> float:
    return intersection_area(prediction, ground_truth) / ground_truth.area if ground_truth.area else 0.0


def class_token(value: str) -> str:
    token = "".join(character for character in str(value or "").lower() if character.isalpha())
    token = token.removesuffix("sku")
    return "handwritten" if token in {"handwritte", "handwritten"} else token


def normalize_points(points: list) -> list[list[int]]:
    normalized = []
    for point in points or []:
        if isinstance(point, dict):
            x = point.get("x", point.get("0"))
            y = point.get("y", point.get("1"))
        else:
            x, y = point[:2]
        if x is not None and y is not None:
            normalized.append([int(round(float(x))), int(round(float(y)))])
    return normalized


def load_truth_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ground_truth_polygons(dataset: Path, rows: list[dict]) -> dict[str, list[list[int]]]:
    by_json: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_json[str(row.get("json_file", ""))].append(row)
    polygons: dict[str, list[list[int]]] = {}
    for json_name, json_rows in by_json.items():
        json_path = dataset / json_name
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = payload.get("shapes", [])
        for row in json_rows:
            try:
                shape = shapes[int(row.get("shape_idx", -1))]
            except (IndexError, TypeError, ValueError):
                continue
            polygons[str(row.get("region_key", ""))] = normalize_points(shape.get("points") or [])
    return polygons


def load_od_ground_truth(dataset: Path, image_names: list[str]) -> dict[str, list[dict]]:
    """Load parent-object polygons used to evaluate raw object detection."""
    target_classes = {"pallet", "rdc", "printedonbox", "handwritten", "other"}
    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    for image_name in image_names:
        json_path = dataset / f"{Path(image_name).stem}.json"
        if not json_path.exists():
            continue
        shapes = json.loads(json_path.read_text(encoding="utf-8")).get("shapes", [])
        for shape_idx, shape in enumerate(shapes):
            label = str(shape.get("label", ""))
            if label.lower().endswith("_sku") or class_token(label) not in target_classes:
                continue
            points = normalize_points(shape.get("points") or [])
            if len(points) < 2:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            box = Box(min(xs), min(ys), max(xs), max(ys))
            if box.area <= 0:
                continue
            rows_by_image[image_name].append(
                {
                    "region_key": f"{image_name}|OD|{shape_idx}",
                    "image": image_name,
                    "shape_idx": shape_idx,
                    "label": label,
                    "bbox_x1": box.x1,
                    "bbox_y1": box.y1,
                    "bbox_x2": box.x2,
                    "bbox_y2": box.y2,
                }
            )
    return rows_by_image


def center_in_box(ground_truth: Box, prediction: Box) -> bool:
    center_x = (ground_truth.x1 + ground_truth.x2) / 2
    center_y = (ground_truth.y1 + ground_truth.y2) / 2
    return prediction.x1 <= center_x < prediction.x2 and prediction.y1 <= center_y < prediction.y2


def match_sku_regions_to_tracks(
    predictions: list[dict],
    sku_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """One-to-one class-aware assignment by SKU center inside buffered crop."""
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        prediction_box = box_from_dict(prediction.get("buffered_bbox"))
        raw_box = box_from_dict(prediction.get("orig_bbox"))
        if prediction_box is None:
            continue
        for gt_index, gt_row in enumerate(sku_rows):
            gt_box = box_from_row(gt_row)
            if (
                gt_box is None
                or class_token(prediction.get("class_name", ""))
                != class_token(gt_row.get("label", ""))
                or not center_in_box(gt_box, prediction_box)
            ):
                continue
            candidates.append(
                (
                    float(prediction.get("confidence", 0) or 0),
                    box_iou(raw_box, gt_box) if raw_box else 0.0,
                    prediction_index,
                    gt_index,
                )
            )
    used_predictions = set()
    used_ground_truth = set()
    matches = []
    for confidence, overlap, prediction_index, gt_index in sorted(
        candidates,
        reverse=True,
    ):
        if prediction_index in used_predictions or gt_index in used_ground_truth:
            continue
        used_predictions.add(prediction_index)
        used_ground_truth.add(gt_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "gt_index": gt_index,
                "iou": overlap,
            }
        )
    surplus = [
        {"prediction_index": index}
        for index in range(len(predictions))
        if index not in used_predictions
    ]
    misses = [
        {"gt_index": index}
        for index in range(len(sku_rows))
        if index not in used_ground_truth
    ]
    return matches, surplus, misses


def segmentation_polygons(track: dict, block_name: str) -> list[list[list[int]]]:
    geometry = (
        track.get("segmentation", {})
        .get(block_name, {})
        .get("original_image", {})
    )
    polygons = geometry.get("polygons") or []
    if polygons:
        return [[normalize_points(polygon) for polygon in polygons][index] for index in range(len(polygons))]
    return [
        normalize_points(rect.get("box_points") or [])
        for rect in geometry.get("rotated_rects") or []
        if rect.get("box_points")
    ]


def polygon_metrics(
    ground_truth_polygon: list[list[int]],
    predicted_polygons: list[list[list[int]]],
) -> tuple[float, float, float]:
    """Return IoU, GT coverage, and prediction precision for region polygons."""
    valid_predictions = [polygon for polygon in predicted_polygons if len(polygon) >= 3]
    if len(ground_truth_polygon) < 3 or not valid_predictions:
        return 0.0, 0.0, 0.0
    all_points = [*ground_truth_polygon]
    for polygon in valid_predictions:
        all_points.extend(polygon)
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    width, height = x2 - x1 + 3, y2 - y1 + 3
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    pred_mask = np.zeros_like(gt_mask)
    gt_points = np.array(
        [[point[0] - x1 + 1, point[1] - y1 + 1] for point in ground_truth_polygon],
        dtype=np.int32,
    )
    cv2.fillPoly(gt_mask, [gt_points], 1)
    for polygon in valid_predictions:
        points = np.array(
            [[point[0] - x1 + 1, point[1] - y1 + 1] for point in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(pred_mask, [points], 1)
    intersection = int(np.logical_and(gt_mask, pred_mask).sum())
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    return (
        intersection / union if union else 0.0,
        intersection / gt_area if gt_area else 0.0,
        intersection / pred_area if pred_area else 0.0,
    )


def greedy_one_to_one_matches(
    predictions: list[dict],
    ground_truth_rows: list[dict],
    bbox_field: str,
    iou_threshold: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Confidence-ordered one-to-one matching, consistent with detector evaluation."""
    unmatched_gt = set(range(len(ground_truth_rows)))
    matches: list[dict] = []
    false_positives: list[dict] = []
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: float(predictions[index].get("confidence", 0) or 0),
        reverse=True,
    )
    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        prediction_box = box_from_dict(prediction.get(bbox_field))
        all_candidates = []
        if prediction_box:
            for gt_index, gt_row in enumerate(ground_truth_rows):
                if class_token(prediction.get("class_name", "")) != class_token(
                    gt_row.get("label", "")
                ):
                    continue
                gt_box = box_from_row(gt_row)
                if gt_box:
                    all_candidates.append((box_iou(prediction_box, gt_box), gt_index))
        candidates = [
            candidate for candidate in all_candidates if candidate[1] in unmatched_gt
        ]
        best_iou, best_gt_index = max(candidates, default=(0.0, -1))
        if best_gt_index >= 0 and best_iou >= iou_threshold:
            unmatched_gt.remove(best_gt_index)
            matches.append(
                {
                    "prediction_index": prediction_index,
                    "gt_index": best_gt_index,
                    "iou": best_iou,
                }
            )
        else:
            false_positives.append(
                {
                    "prediction_index": prediction_index,
                    "best_iou": max(
                        (candidate[0] for candidate in all_candidates),
                        default=0.0,
                    ),
                }
            )
    false_negatives = [{"gt_index": index} for index in sorted(unmatched_gt)]
    return matches, false_positives, false_negatives


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calculate_pr(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def draw_failure_overlay(
    image_path: Path,
    output_path: Path,
    ground_truth_rows: list[dict],
    predictions: list[dict],
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for row in ground_truth_rows:
        box = box_from_row(row)
        if box:
            cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)
    for prediction in predictions:
        raw = box_from_dict(prediction.get("orig_bbox"))
        buffered = box_from_dict(prediction.get("buffered_bbox"))
        if raw:
            cv2.rectangle(image, (raw.x1, raw.y1), (raw.x2, raw.y2), (0, 180, 0), 2)
        if buffered:
            cv2.rectangle(
                image,
                (buffered.x1, buffered.y1),
                (buffered.x2, buffered.y2),
                (255, 180, 0),
                1,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run_evaluation(args: argparse.Namespace) -> dict:
    truth_rows = load_truth_rows(args.truth_csv)
    truth_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in truth_rows:
        truth_by_image[str(row.get("image", ""))].append(row)
    truth_polygons = load_ground_truth_polygons(args.dataset, truth_rows)
    prediction_images = available_prediction_images(args.predictions_dir)
    evaluation_images = sorted(set(prediction_images) | set(truth_by_image))
    od_truth_by_image = load_od_ground_truth(args.dataset, evaluation_images)

    detection_details: list[dict] = []
    pipeline_details: list[dict] = []
    image_metrics: list[dict] = []
    class_counts = defaultdict(Counter)
    failure_image_scores: Counter = Counter()

    total_predictions = 0
    total_seg_found = 0
    total_ocr_found = 0
    total_valid_ocr = 0
    total_duplicate_det_ids = 0

    for image_name in evaluation_images:
        image_sku_truth = truth_by_image.get(image_name, [])
        image_od_truth = od_truth_by_image.get(image_name, [])
        predictions = load_prediction_tracks_for_image(
            args.predictions_dir,
            image_name,
            truth_rows,
        )
        total_predictions += len(predictions)
        total_seg_found += sum(bool(item.get("seg_found")) for item in predictions)
        total_ocr_found += sum(bool(item.get("ocr_found")) for item in predictions)
        total_valid_ocr += sum(bool(prediction_text(item)) for item in predictions)
        det_id_counts = Counter(
            str(item.get("det_id"))
            for item in predictions
            if item.get("det_id") not in {None, ""}
        )
        duplicate_det_ids = sum(count - 1 for count in det_id_counts.values() if count > 1)
        total_duplicate_det_ids += duplicate_det_ids
        per_image = {
            "image": image_name,
            "sku_ground_truth_count": len(image_sku_truth),
            "od_ground_truth_count": len(image_od_truth),
            "prediction_count": len(predictions),
            "segmentation_found_count": sum(bool(item.get("seg_found")) for item in predictions),
            "ocr_found_count": sum(bool(item.get("ocr_found")) for item in predictions),
            "valid_sku_count": sum(bool(prediction_text(item)) for item in predictions),
            "duplicate_det_id_count": duplicate_det_ids,
        }

        od_matches, false_positives, false_negatives = greedy_one_to_one_matches(
            predictions,
            image_od_truth,
            "orig_bbox",
            args.iou_threshold,
        )
        per_image["detection_tp"] = len(od_matches)
        per_image["detection_fp"] = len(false_positives)
        per_image["detection_fn"] = len(false_negatives)
        failure_image_scores[image_name] += len(false_positives) + len(false_negatives)

        # Raw OD establishes the identity match. Both boxes are then measured
        # against that same GT region so crop buffering cannot change TP/FP/FN.
        for geometry_name, bbox_field in (
            ("raw", "orig_bbox"),
            ("buffered", "buffered_bbox"),
        ):
            for match in od_matches:
                prediction = predictions[match["prediction_index"]]
                gt_row = image_od_truth[match["gt_index"]]
                prediction_box = box_from_dict(prediction.get(bbox_field))
                gt_box = box_from_row(gt_row)
                iou = box_iou(prediction_box, gt_box) if prediction_box and gt_box else 0.0
                coverage = (
                    ground_truth_coverage(prediction_box, gt_box)
                    if prediction_box and gt_box
                    else 0.0
                )
                detail = {
                    "image": image_name,
                    "geometry": geometry_name,
                    "outcome": "TP",
                    "class": class_token(gt_row.get("label", "")),
                    "det_id": prediction.get("det_id"),
                    "confidence": prediction.get("confidence"),
                    "region_key": gt_row.get("region_key"),
                    "iou": iou,
                    "ground_truth_coverage": coverage,
                    "fully_covered": "yes" if coverage >= 0.999 else "no",
                    "failure_category": "",
                }
                detection_details.append(detail)
                if geometry_name == "raw":
                    class_counts[detail["class"]]["tp"] += 1

        for false_positive in false_positives:
            prediction = predictions[false_positive["prediction_index"]]
            predicted_class = class_token(prediction.get("class_name", ""))
            detection_details.append(
                {
                    "image": image_name,
                    "geometry": "raw",
                    "outcome": "FP",
                    "class": predicted_class,
                    "det_id": prediction.get("det_id"),
                    "confidence": prediction.get("confidence"),
                    "region_key": "",
                    "iou": false_positive["best_iou"],
                    "ground_truth_coverage": 0.0,
                    "fully_covered": "no",
                    "failure_category": (
                        "duplicate_prediction"
                        if false_positive["best_iou"] >= args.iou_threshold
                        else "false_positive"
                    ),
                }
            )
            class_counts[predicted_class]["fp"] += 1

        for false_negative in false_negatives:
            gt_row = image_od_truth[false_negative["gt_index"]]
            gt_class = class_token(gt_row.get("label", ""))
            detection_details.append(
                {
                    "image": image_name,
                    "geometry": "raw",
                    "outcome": "FN",
                    "class": gt_class,
                    "det_id": "",
                    "confidence": "",
                    "region_key": gt_row.get("region_key"),
                    "iou": 0.0,
                    "ground_truth_coverage": 0.0,
                    "fully_covered": "no",
                    "failure_category": "detection_missed",
                }
            )
            class_counts[gt_class]["fn"] += 1

        sku_matches, _, sku_misses = match_sku_regions_to_tracks(
            predictions,
            image_sku_truth,
        )
        per_image["sku_track_matches"] = len(sku_matches)
        per_image["sku_track_misses"] = len(sku_misses)
        prediction_by_gt = {
            image_sku_truth[match["gt_index"]].get("region_key"): predictions[
                match["prediction_index"]
            ]
            for match in sku_matches
        }
        for gt_row in image_sku_truth:
            region_key = str(gt_row.get("region_key", ""))
            prediction = prediction_by_gt.get(region_key)
            gt_box = box_from_row(gt_row)
            any_compatible_track_center_hit = bool(
                gt_box
                and any(
                    class_token(candidate.get("class_name", ""))
                    == class_token(gt_row.get("label", ""))
                    and (candidate_box := box_from_dict(candidate.get("buffered_bbox")))
                    and center_in_box(gt_box, candidate_box)
                    for candidate in predictions
                )
            )
            expected_sku = str(gt_row.get("expected_sku", "") or "").strip()
            scorable = is_scorable_review(
                gt_row.get("review_status", ""),
                expected_sku,
                gt_row.get("notes", ""),
            )
            predicted_sku = prediction_text(prediction) if prediction else ""
            raw_box = box_from_dict(prediction.get("orig_bbox")) if prediction else None
            buffered_box = (
                box_from_dict(prediction.get("buffered_bbox")) if prediction else None
            )
            raw_crop_coverage = (
                ground_truth_coverage(raw_box, gt_box) if raw_box and gt_box else 0.0
            )
            buffered_crop_coverage = (
                ground_truth_coverage(buffered_box, gt_box)
                if buffered_box and gt_box
                else 0.0
            )
            raw_seg_iou = raw_seg_coverage = raw_seg_precision = 0.0
            post_seg_iou = post_seg_coverage = post_seg_precision = 0.0
            if prediction:
                gt_polygon = truth_polygons.get(region_key, [])
                raw_seg_iou, raw_seg_coverage, raw_seg_precision = polygon_metrics(
                    gt_polygon,
                    segmentation_polygons(prediction, "raw_prediction"),
                )
                (
                    post_seg_iou,
                    post_seg_coverage,
                    post_seg_precision,
                ) = polygon_metrics(
                    gt_polygon,
                    segmentation_polygons(prediction, "postprocessed_minAreaRect"),
                )

            if not prediction:
                failure_category = "detection_missed"
            elif buffered_crop_coverage < 0.9:
                failure_category = "poor_crop_coverage"
            elif not prediction.get("seg_found"):
                failure_category = "segmentation_missing"
            elif not scorable:
                failure_category = "not_ocr_scorable"
            elif not predicted_sku and prediction.get("ocr_found"):
                failure_category = "invalid_sku_length"
            elif not predicted_sku:
                failure_category = "ocr_missing"
            elif normalize_sku_digits(expected_sku) != normalize_sku_digits(predicted_sku):
                failure_category = "ocr_incorrect"
            else:
                failure_category = "correct"
            if failure_category not in {"correct", "not_ocr_scorable"}:
                failure_image_scores[image_name] += 1

            pipeline_details.append(
                {
                    "image": image_name,
                    "region_key": region_key,
                    "class": class_token(gt_row.get("label", "")),
                    "expected_sku": expected_sku,
                    "sku_digit_class": sku_digit_class(expected_sku),
                    "scorable": "yes" if scorable else "no",
                    "any_compatible_track_center_hit": (
                        "yes" if any_compatible_track_center_hit else "no"
                    ),
                    "detected": "yes" if prediction else "no",
                    "det_id": prediction.get("det_id") if prediction else "",
                    "seg_found": "yes" if prediction and prediction.get("seg_found") else "no",
                    "ocr_found": "yes" if prediction and prediction.get("ocr_found") else "no",
                    "predicted_sku": predicted_sku,
                    "raw_crop_gt_coverage": raw_crop_coverage,
                    "buffered_crop_gt_coverage": buffered_crop_coverage,
                    "ocr_exact_match": (
                        "yes"
                        if scorable
                        and predicted_sku
                        and normalize_sku_digits(expected_sku)
                        == normalize_sku_digits(predicted_sku)
                        else "no" if scorable else "n/a"
                    ),
                    "raw_segmentation_iou_proxy": raw_seg_iou,
                    "raw_segmentation_gt_coverage": raw_seg_coverage,
                    "raw_segmentation_precision_proxy": raw_seg_precision,
                    "post_segmentation_iou_proxy": post_seg_iou,
                    "post_segmentation_gt_coverage": post_seg_coverage,
                    "post_segmentation_precision_proxy": post_seg_precision,
                    "failure_category": failure_category,
                }
            )
        image_metrics.append(per_image)

    class_metric_rows = []
    for class_name, counts in sorted(class_counts.items()):
        precision, recall, f1 = calculate_pr(
            counts["tp"],
            counts["fp"],
            counts["fn"],
        )
        class_metric_rows.append(
            {
                "class": class_name,
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    summary_rows = []

    def add_metric(name: str, value, denominator="", notes=""):
        summary_rows.append(
            {
                "metric": name,
                "value": value,
                "denominator": denominator,
                "notes": notes,
            }
        )

    add_metric("images_evaluated", len(evaluation_images))
    add_metric("sku_ground_truth_regions", len(truth_rows))
    add_metric(
        "od_ground_truth_regions",
        sum(len(rows) for rows in od_truth_by_image.values()),
    )
    add_metric("predictions", total_predictions)
    add_metric(
        "empty_prediction_images",
        sum(row["prediction_count"] == 0 for row in image_metrics),
        len(evaluation_images),
    )
    add_metric("duplicate_det_ids", total_duplicate_det_ids, total_predictions)
    add_metric(
        "segmentation_found_rate",
        total_seg_found / total_predictions if total_predictions else 0.0,
        total_predictions,
    )
    add_metric(
        "ocr_found_rate",
        total_ocr_found / total_predictions if total_predictions else 0.0,
        total_predictions,
    )
    add_metric(
        "valid_6_or_10_digit_sku_rate",
        total_valid_ocr / total_predictions if total_predictions else 0.0,
        total_predictions,
    )
    add_metric(
        "duplicate_overlapping_predictions",
        sum(
            row["failure_category"] == "duplicate_prediction"
            for row in detection_details
        ),
        total_predictions,
        "Same-class FP overlapping an already matched GT at the IoU threshold",
    )
    raw_rows = [row for row in detection_details if row["geometry"] == "raw"]
    outcomes = Counter(row["outcome"] for row in raw_rows)
    precision, recall, f1 = calculate_pr(
        outcomes["TP"],
        outcomes["FP"],
        outcomes["FN"],
    )
    add_metric(f"detection_precision_at_iou_{args.iou_threshold}", precision)
    add_metric(f"detection_recall_at_iou_{args.iou_threshold}", recall)
    add_metric(f"detection_f1_at_iou_{args.iou_threshold}", f1)
    for geometry_name in ("raw", "buffered"):
        tp_rows = [
            row
            for row in detection_details
            if row["geometry"] == geometry_name and row["outcome"] == "TP"
        ]
        add_metric(f"od_{geometry_name}_mean_matched_iou", safe_mean([row["iou"] for row in tp_rows]))
        add_metric(
            f"od_{geometry_name}_mean_gt_coverage",
            safe_mean([row["ground_truth_coverage"] for row in tp_rows]),
        )
        add_metric(
            f"od_{geometry_name}_full_gt_coverage_rate",
            safe_mean([1.0 if row["fully_covered"] == "yes" else 0.0 for row in tp_rows]),
        )

    scorable_rows = [row for row in pipeline_details if row["scorable"] == "yes"]
    exact_rows = [row for row in scorable_rows if row["ocr_exact_match"] == "yes"]
    detected_scorable = [row for row in scorable_rows if row["detected"] == "yes"]
    predicted_scorable = [row for row in detected_scorable if row["predicted_sku"]]
    ci_low, ci_high = wilson_interval(len(exact_rows), len(scorable_rows))
    add_metric(
        "interim_end_to_end_ocr_exact_accuracy",
        len(exact_rows) / len(scorable_rows) if scorable_rows else 0.0,
        len(scorable_rows),
        "Preliminary until manual annotation is complete",
    )
    add_metric("interim_ocr_accuracy_ci95_low", ci_low, len(scorable_rows))
    add_metric("interim_ocr_accuracy_ci95_high", ci_high, len(scorable_rows))
    add_metric(
        "interim_conditional_ocr_exact_accuracy",
        len(exact_rows) / len(predicted_scorable) if predicted_scorable else 0.0,
        len(predicted_scorable),
        "Among scorable GT regions with a valid OCR prediction",
    )
    add_metric(
        "scorable_any_track_center_coverage_rate",
        safe_mean(
            [
                1.0 if row["any_compatible_track_center_hit"] == "yes" else 0.0
                for row in scorable_rows
            ]
        ),
        len(scorable_rows),
        "Coverage before strict one-to-one assignment",
    )
    add_metric(
        "scorable_sku_track_match_rate",
        len(detected_scorable) / len(scorable_rows) if scorable_rows else 0.0,
        len(scorable_rows),
    )
    add_metric(
        "scorable_buffered_full_coverage_rate",
        safe_mean(
            [
                1.0 if float(row["buffered_crop_gt_coverage"]) >= 0.999 else 0.0
                for row in detected_scorable
            ]
        ),
        len(detected_scorable),
    )
    add_metric(
        "scorable_buffered_mean_gt_coverage",
        safe_mean(
            [float(row["buffered_crop_gt_coverage"]) for row in detected_scorable]
        ),
        len(detected_scorable),
    )

    for block_prefix in ("raw", "post"):
        matched_pipeline = [
            row for row in pipeline_details if row["detected"] == "yes"
        ]
        add_metric(
            f"{block_prefix}_segmentation_mean_iou_proxy",
            safe_mean(
                [
                    float(row[f"{block_prefix}_segmentation_iou_proxy"])
                    for row in matched_pipeline
                ]
            ),
            len(matched_pipeline),
            "Region proxy; only true mask IoU if GT polygons are segmentation masks",
        )
        add_metric(
            f"{block_prefix}_segmentation_mean_gt_coverage",
            safe_mean(
                [
                    float(row[f"{block_prefix}_segmentation_gt_coverage"])
                    for row in matched_pipeline
                ]
            ),
            len(matched_pipeline),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "class_metrics.csv", class_metric_rows)
    write_csv(args.output_dir / "detection_details.csv", detection_details)
    write_csv(args.output_dir / "pipeline_details.csv", pipeline_details)
    write_csv(args.output_dir / "image_metrics.csv", image_metrics)

    failure_counts = Counter(row["failure_category"] for row in pipeline_details)
    detection_failure_counts = Counter(
        row["failure_category"]
        for row in detection_details
        if row["failure_category"]
    )
    report_lines = [
        "# Overhead Pipeline Evaluation",
        "",
        f"- Images evaluated: {len(evaluation_images)}",
        f"- SKU ground-truth regions: {len(truth_rows)}",
        f"- OD ground-truth regions: {sum(len(rows) for rows in od_truth_by_image.values())}",
        f"- Predictions: {total_predictions}",
        f"- IoU threshold: {args.iou_threshold}",
        "",
        "## Summary metrics",
        "",
    ]
    for row in summary_rows:
        value = row["value"]
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        report_lines.append(f"- **{row['metric']}**: {rendered}")
    report_lines.extend(["", "## Failure categories", ""])
    for category, count in sorted(failure_counts.items()):
        report_lines.append(f"- **{category}**: {count}")
    report_lines.extend(["", "## Detection-level failures", ""])
    for category, count in sorted(detection_failure_counts.items()):
        report_lines.append(f"- **{category}**: {count}")
    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Segmentation metrics are region proxies unless the ground-truth polygons are confirmed masks.",
            "- OCR metrics are interim and automatically update as more expected SKUs are reviewed.",
            "- OD matches parent-object GT by raw-box IoU; SKU stages match label centers inside buffered crops.",
            "- Both matching planes are class-aware and one-to-one.",
            "- Any-track center coverage is also reported before one-to-one collision resolution.",
            "",
        ]
    )
    (args.output_dir / "evaluation_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    failure_overlay_dir = args.output_dir / "failure_overlays"
    for image_name, _ in failure_image_scores.most_common(args.max_failure_overlays):
        draw_failure_overlay(
            args.dataset / image_name,
            failure_overlay_dir / image_name,
            truth_by_image.get(image_name, []),
            load_prediction_tracks_for_image(args.predictions_dir, image_name, truth_rows),
        )

    return {
        "summary_rows": summary_rows,
        "failure_counts": failure_counts,
        "output_dir": args.output_dir,
    }


def main() -> None:
    args = parse_args()
    result = run_evaluation(args)
    print(f"Wrote evaluation artifacts to {result['output_dir']}")
    for row in result["summary_rows"]:
        value = row["value"]
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"{row['metric']}: {rendered}")


if __name__ == "__main__":
    main()

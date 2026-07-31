#!/usr/bin/env python3
"""Compare saved local predictions with TorchServe on the same golden inputs."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

IOU_THRESHOLDS = (0.50, 0.75, 0.90, 0.95)


def box_iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 1.0


def match_detections(
    local: list[dict[str, Any]], served: list[list[float]]
) -> list[tuple[int, int, float]]:
    if not local or not served:
        return []
    candidates: list[tuple[float, int, int]] = []
    for local_index, local_item in enumerate(local):
        for served_index, served_item in enumerate(served):
            if int(local_item["class_id"]) == int(served_item[5]):
                iou = box_iou(local_item["bbox"], served_item[:4])
                if iou > 0.0:
                    candidates.append((iou, local_index, served_index))
    matched_local: set[int] = set()
    matched_served: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, local_index, served_index in sorted(candidates, reverse=True):
        if local_index in matched_local or served_index in matched_served:
            continue
        matched_local.add(local_index)
        matched_served.add(served_index)
        matches.append((local_index, served_index, iou))
    return matches


def detection_metrics(
    image_name: str,
    local_records: list[dict[str, Any]],
    served: list[list[float]],
    latency_ms: float,
) -> dict[str, Any]:
    local = [
        {
            "class_id": int(item["class_id"]),
            "confidence": float(item["confidence"]),
            "bbox": [
                float(item["orig_bbox"]["x1"]),
                float(item["orig_bbox"]["y1"]),
                float(item["orig_bbox"]["x2"]),
                float(item["orig_bbox"]["y2"]),
            ],
        }
        for item in local_records
    ]
    matches = match_detections(local, served)
    row: dict[str, Any] = {
        "image": image_name,
        "local_count": len(local),
        "torchserve_count": len(served),
        "latency_ms": round(latency_ms, 3),
    }
    for threshold in IOU_THRESHOLDS:
        matched = sum(iou >= threshold for _, _, iou in matches)
        suffix = str(int(threshold * 100))
        row[f"matched_iou_{suffix}"] = matched
        row[f"precision_iou_{suffix}"] = matched / len(served) if served else float(not local)
        row[f"recall_iou_{suffix}"] = matched / len(local) if local else float(not served)

    matched_50 = [(a, b, iou) for a, b, iou in matches if iou >= 0.50]
    row["mean_matched_iou"] = (
        statistics.fmean(iou for _, _, iou in matched_50) if matched_50 else 0.0
    )
    row["min_matched_iou"] = min((iou for _, _, iou in matched_50), default=0.0)
    row["max_bbox_delta_px"] = max(
        (
            abs(local[local_index]["bbox"][axis] - served[served_index][axis])
            for local_index, served_index, _ in matched_50
            for axis in range(4)
        ),
        default=0.0,
    )
    row["max_confidence_delta"] = max(
        (
            abs(
                local[local_index]["confidence"]
                - float(served[served_index][4])
            )
            for local_index, served_index, _ in matched_50
        ),
        default=0.0,
    )
    row["within_one_pixel_and_0_001_conf"] = sum(
        max(
            abs(local[local_index]["bbox"][axis] - served[served_index][axis])
            for axis in range(4)
        )
        <= 1.0
        and abs(
            local[local_index]["confidence"] - float(served[served_index][4])
        )
        <= 0.001
        for local_index, served_index, _ in matched_50
    )
    return row


def decode_mask(mask_payload: dict[str, Any] | None, fallback_shape: tuple[int, int]) -> np.ndarray:
    if not mask_payload or not mask_payload.get("img"):
        return np.zeros(fallback_shape, dtype=bool)
    decoded = np.array(Image.open(io.BytesIO(base64.b64decode(mask_payload["img"])))) > 127
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded


def filter_served_detections(
    served: list[list[float]],
    min_confidence: float,
) -> list[list[float]]:
    return [row for row in served if float(row[4]) >= min_confidence]


def segmentation_metrics(
    image_name: str,
    strip_index: int,
    local_mask: np.ndarray,
    served_mask: np.ndarray,
    detection_count: int,
    latency_ms: float,
) -> dict[str, Any]:
    if local_mask.shape != served_mask.shape:
        return {
            "image": image_name,
            "strip_index": strip_index,
            "shape_equal": False,
            "local_shape": "x".join(map(str, local_mask.shape)),
            "torchserve_shape": "x".join(map(str, served_mask.shape)),
            "torchserve_detection_count": detection_count,
            "latency_ms": round(latency_ms, 3),
        }
    intersection = int(np.count_nonzero(local_mask & served_mask))
    union = int(np.count_nonzero(local_mask | served_mask))
    local_pixels = int(np.count_nonzero(local_mask))
    served_pixels = int(np.count_nonzero(served_mask))
    different_pixels = int(np.count_nonzero(local_mask != served_mask))
    total_pixels = int(local_mask.size)
    iou = intersection / union if union else 1.0
    denominator = local_pixels + served_pixels
    dice = (2 * intersection / denominator) if denominator else 1.0
    return {
        "image": image_name,
        "strip_index": strip_index,
        "shape_equal": True,
        "height": local_mask.shape[0],
        "width": local_mask.shape[1],
        "local_white_pixels": local_pixels,
        "torchserve_white_pixels": served_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "different_pixels": different_pixels,
        "pixel_agreement": (total_pixels - different_pixels) / total_pixels,
        "mask_iou": iou,
        "dice": dice,
        "torchserve_detection_count": detection_count,
        "latency_ms": round(latency_ms, 3),
    }


def summarize(
    detection_rows: list[dict[str, Any]],
    segmentation_rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "detection_images": len(detection_rows),
        "segmentation_strips": len(segmentation_rows),
        "failures": failures,
    }
    if detection_rows:
        local_total = sum(row["local_count"] for row in detection_rows)
        served_total = sum(row["torchserve_count"] for row in detection_rows)
        detection: dict[str, Any] = {
            "local_detections": local_total,
            "torchserve_detections": served_total,
            "mean_matched_iou": statistics.fmean(
                row["mean_matched_iou"] for row in detection_rows
            ),
            "mean_latency_ms": statistics.fmean(
                row["latency_ms"] for row in detection_rows
            ),
            "max_bbox_delta_px": max(
                row["max_bbox_delta_px"] for row in detection_rows
            ),
            "max_confidence_delta": max(
                row["max_confidence_delta"] for row in detection_rows
            ),
            "near_exact_matches": sum(
                row["within_one_pixel_and_0_001_conf"] for row in detection_rows
            ),
        }
        for threshold in IOU_THRESHOLDS:
            suffix = str(int(threshold * 100))
            matched = sum(row[f"matched_iou_{suffix}"] for row in detection_rows)
            precision = matched / served_total if served_total else float(not local_total)
            recall = matched / local_total if local_total else float(not served_total)
            detection[f"matched_iou_{suffix}"] = matched
            detection[f"precision_iou_{suffix}"] = precision
            detection[f"recall_iou_{suffix}"] = recall
            detection[f"f1_iou_{suffix}"] = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        summary["detection"] = detection

    valid_masks = [
        row for row in segmentation_rows if row.get("shape_equal") is True
    ]
    if valid_masks:
        summary["segmentation"] = {
            "shape_matches": len(valid_masks),
            "shape_mismatches": len(segmentation_rows) - len(valid_masks),
            "mean_mask_iou": statistics.fmean(row["mask_iou"] for row in valid_masks),
            "median_mask_iou": statistics.median(
                row["mask_iou"] for row in valid_masks
            ),
            "minimum_mask_iou": min(row["mask_iou"] for row in valid_masks),
            "mean_dice": statistics.fmean(row["dice"] for row in valid_masks),
            "mean_pixel_agreement": statistics.fmean(
                row["pixel_agreement"] for row in valid_masks
            ),
            "exact_masks": sum(row["different_pixels"] == 0 for row in valid_masks),
            "iou_at_least_0_95": sum(row["mask_iou"] >= 0.95 for row in valid_masks),
            "iou_at_least_0_90": sum(row["mask_iou"] >= 0.90 for row in valid_masks),
            "mean_latency_ms": statistics.fmean(
                row["latency_ms"] for row in valid_masks
            ),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-images", type=Path, required=True)
    parser.add_argument("--local-outputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector-url", required=True)
    parser.add_argument("--segmenter-url", required=True)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="Filter TorchServe detections to match local ground-truth confidence floor.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.golden_images.glob("*.jpg"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    session = requests.Session()
    detection_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for image_number, image_path in enumerate(image_paths, start=1):
        local_dir = args.local_outputs / image_path.stem
        predictions_path = local_dir / "predictions.json"
        if not predictions_path.is_file():
            failures.append({"image": image_path.name, "error": "predictions.json missing"})
            continue
        try:
            local_records = json.loads(predictions_path.read_text())
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            started = time.perf_counter()
            response = session.post(
                args.detector_url,
                json={
                    "instances": [
                        {"model_name": "detection", "file": image_base64}
                    ]
                },
                timeout=180,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()
            if "error" in payload and "predictions" not in payload:
                raise RuntimeError(payload["error"])
            served_raw = payload["predictions"][0]["detections"]
            served = filter_served_detections(served_raw, args.min_confidence)
            row = detection_metrics(
                image_path.name, local_records, served, latency_ms
            )
            row["torchserve_raw_count"] = len(served_raw)
            row["min_confidence_filter"] = args.min_confidence
            detection_rows.append(row)

            strip_paths = sorted(
                (local_dir / "strips").glob("strip_*.jpg"),
                key=lambda path: int(path.stem.rsplit("_", 1)[1]),
            )
            if strip_paths:
                instances = [
                    {
                        "model_name": "segmentation",
                        "strip_id": int(path.stem.rsplit("_", 1)[1]),
                        "file": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                    for path in strip_paths
                ]
                started = time.perf_counter()
                response = session.post(
                    args.segmenter_url,
                    json={"instances": instances},
                    timeout=180,
                )
                segmentation_latency_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()
                if "error" in payload and "predictions" not in payload:
                    raise RuntimeError(payload["error"])
                served_results = payload["predictions"][0]["segmentation"][
                    "seg_results"
                ]
                latency_per_strip = segmentation_latency_ms / len(strip_paths)
                for served_result in served_results:
                    strip_index = int(served_result["strip_index"])
                    mask_path = (
                        local_dir
                        / "strips_viz_orig_mask"
                        / f"{strip_index}_pred_mask.jpg"
                    )
                    if not mask_path.is_file():
                        failures.append(
                            {
                                "image": image_path.name,
                                "error": f"local mask missing for strip {strip_index}",
                            }
                        )
                        continue
                    local_mask = np.array(Image.open(mask_path).convert("L")) > 127
                    served_mask = decode_mask(
                        served_result.get("mask"), local_mask.shape
                    )
                    segmentation_rows.append(
                        segmentation_metrics(
                            image_path.name,
                            strip_index,
                            local_mask,
                            served_mask,
                            len(served_result.get("detections") or []),
                            latency_per_strip,
                        )
                    )
        except Exception as exc:  # continue to produce a complete failure report
            failures.append({"image": image_path.name, "error": str(exc)})

        print(
            f"[{image_number}/{len(image_paths)}] {image_path.name} "
            f"detection_rows={len(detection_rows)} "
            f"segmentation_rows={len(segmentation_rows)} failures={len(failures)}",
            flush=True,
        )

    summary = summarize(detection_rows, segmentation_rows, failures)
    summary["comparison"] = {
        "local_role": "ground_truth",
        "torchserve_role": "experimental",
        "min_confidence_filter": args.min_confidence,
        "detector_url": args.detector_url,
        "segmenter_url": args.segmenter_url,
        "notes": [
            "Local detection ground truth comes from predictions.json (conf floor ~0.25).",
            "TorchServe detections are filtered with --min-confidence before IoU matching.",
            "Local segmentation masks are JPEG exports and may lose exact pixel fidelity.",
        ],
    }
    write_csv(output_dir / "detection_per_image.csv", detection_rows)
    write_csv(output_dir / "segmentation_per_strip.csv", segmentation_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

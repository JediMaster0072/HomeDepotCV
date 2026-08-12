#!/usr/bin/env python3
"""Raw-vs-raw segmentation exactness: in-process Stage2 vs TorchServe.

Uses the same segmentation.pt and config as the Dockerfile packaging:
  imgsz=608, conf=0.25, iou=0.70, max_det=300, half=False
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image


def decode_served_mask(mask_payload: dict[str, Any] | None, shape: tuple[int, int]) -> np.ndarray:
    if not mask_payload or not mask_payload.get("img"):
        return np.zeros(shape, dtype=bool)
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(mask_payload["img"])))) > 127
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def mask_metrics(local_mask: np.ndarray, served_mask: np.ndarray) -> dict[str, Any]:
    if local_mask.shape != served_mask.shape:
        return {
            "shape_equal": False,
            "local_shape": "x".join(map(str, local_mask.shape)),
            "torchserve_shape": "x".join(map(str, served_mask.shape)),
        }
    intersection = int(np.count_nonzero(local_mask & served_mask))
    union = int(np.count_nonzero(local_mask | served_mask))
    different = int(np.count_nonzero(local_mask != served_mask))
    local_pixels = int(np.count_nonzero(local_mask))
    served_pixels = int(np.count_nonzero(served_mask))
    total = int(local_mask.size)
    return {
        "shape_equal": True,
        "local_white_pixels": local_pixels,
        "torchserve_white_pixels": served_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "different_pixels": different,
        "exact_mask": different == 0,
        "mask_iou": (intersection / union) if union else 1.0,
        "pixel_agreement": (total - different) / total if total else 1.0,
    }


def collect_strips(strips_root: Path, limit: int | None) -> list[Path]:
    paths = sorted(strips_root.glob("**/strip_*.jpg"))
    if not paths:
        paths = sorted(strips_root.glob("strip_*.jpg"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_local_stage(seg_package: Path):
    sys.path.insert(0, str(seg_package))
    from common_config_gpu import build_gpu_config
    from service_pipeline_gpu.label_record import StripInfo
    from service_pipeline_gpu.stage2_segmentation import Stage2Segmentation

    config = build_gpu_config(model_dir=str(seg_package))
    stage = Stage2Segmentation(config)
    stage.load_model()
    return stage, StripInfo, {
        "seg_imgsz": config["seg_imgsz"],
        "seg_conf_thres": config["seg_conf_thres"],
        "seg_iou_thres": config["seg_iou_thres"],
        "seg_max_det": config["seg_max_det"],
        "weights": config["segmentation_weights"],
        "device": str(stage.device),
        "half": stage.half,
        "resolved_imgsz": stage.imgsz,
    }


def encode_png_base64(image_bgr: np.ndarray) -> str:
    """Match the TorchServe client contract: lossless PNG bytes, base64 text."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_handler_image(image_base64: str) -> np.ndarray:
    """Current codec path: PIL RGB for Stage1/Stage2 (no channel flip)."""
    image_bytes = base64.b64decode(image_base64.encode("ascii"))
    with io.BytesIO(image_bytes) as buffer:
        image = Image.open(buffer)
        image.load()
    return np.array(image)


def query_torchserve(endpoint: str, image_base64: str, strip_id: int) -> dict[str, Any]:
    payload = {
        "instances": [
            {
                "model_name": "segmentation",
                "strip_id": strip_id,
                "file": image_base64,
            }
        ]
    }
    response = requests.post(endpoint, json=payload, timeout=180)
    response.raise_for_status()
    body = response.json()
    if "error" in body and "predictions" not in body:
        raise RuntimeError(body["error"])
    return body["predictions"][0]["segmentation"]["seg_results"][0]


def compare_one(
    stage,
    StripInfo,
    endpoint: str,
    strip_path: Path,
    color_mode: str,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(strip_path))
    if image_bgr is None:
        raise FileNotFoundError(f"could not read {strip_path}")
    strip_id = int(strip_path.stem.rsplit("_", 1)[1])
    image_base64 = encode_png_base64(image_bgr)

    if color_mode == "bgr":
        # OpenCV load converted to RGB — Stage expects RGB (no internal flip).
        local_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif color_mode == "handler":
        # Exact array current TorchServe preprocess feeds into Stage2 (RGB).
        local_image = decode_handler_image(image_base64)
    elif color_mode == "legacy_bgr":
        # Raw OpenCV BGR into Stage that expects RGB (wrong channel order).
        local_image = image_bgr
    else:
        raise ValueError(f"unknown color_mode={color_mode}")

    started = time.perf_counter()
    local_result = stage.run_inference(
        [StripInfo(strip_index=strip_id, strip_image=local_image, label_records=[])]
    )[0]
    local_ms = (time.perf_counter() - started) * 1000
    local_mask = decode_served_mask(local_result.get("mask"), image_bgr.shape[:2])

    started = time.perf_counter()
    served = query_torchserve(endpoint, image_base64, strip_id)
    served_ms = (time.perf_counter() - started) * 1000
    served_mask = decode_served_mask(served.get("mask"), image_bgr.shape[:2])

    metrics = mask_metrics(local_mask, served_mask)
    local_dets = local_result.get("detections") or []
    served_dets = served.get("detections") or []
    if len(local_dets) == len(served_dets) and local_dets:
        bbox_delta = max(
            abs(float(a["bbox"][i]) - float(b["bbox"][i]))
            for a, b in zip(local_dets, served_dets)
            for i in range(4)
        )
        conf_delta = max(
            abs(float(a["confidence"]) - float(b["confidence"]))
            for a, b in zip(local_dets, served_dets)
        )
        classes_equal = [int(a["class_id"]) for a in local_dets] == [
            int(b["class_id"]) for b in served_dets
        ]
    else:
        bbox_delta = None
        conf_delta = None
        classes_equal = False

    return {
        "strip_path": str(strip_path),
        "strip_index": strip_id,
        "color_mode": color_mode,
        "local_detection_count": len(local_dets),
        "torchserve_detection_count": len(served_dets),
        "local_latency_ms": round(local_ms, 3),
        "torchserve_latency_ms": round(served_ms, 3),
        "detections_count_equal": len(local_dets) == len(served_dets),
        "detections_classes_equal": classes_equal,
        "max_bbox_delta": bbox_delta,
        "max_conf_delta": conf_delta,
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seg-package",
        type=Path,
        required=True,
        help="Path to cv-singleline-torchserve-dual/segmentation",
    )
    parser.add_argument(
        "--strips-root",
        type=Path,
        required=True,
        help="Directory containing strip_*.jpg files (or nested image folders)",
    )
    parser.add_argument("--segmenter-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--color-mode",
        choices=("bgr", "handler", "legacy_bgr"),
        default="handler",
        help=(
            "Local Stage2 input color handling. "
            "'handler' feeds the current codec RGB array TorchServe uses; "
            "'bgr' uses cv2.imread converted BGR→RGB; "
            "'legacy_bgr' feeds raw OpenCV BGR into Stage that expects RGB."
        ),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    strip_paths = collect_strips(args.strips_root, args.limit)
    if not strip_paths:
        raise FileNotFoundError(f"no strip_*.jpg under {args.strips_root}")

    print(f"loading local Stage2 from {args.seg_package}", flush=True)
    stage, StripInfo, config_info = load_local_stage(args.seg_package.resolve())
    print(
        json.dumps(
            {"local_config": config_info, "color_mode": args.color_mode},
            indent=2,
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, strip_path in enumerate(strip_paths, start=1):
        try:
            rows.append(
                compare_one(
                    stage,
                    StripInfo,
                    args.segmenter_url,
                    strip_path,
                    args.color_mode,
                )
            )
        except Exception as exc:  # keep going for a full report
            failures.append({"strip_path": str(strip_path), "error": str(exc)})

        print(
            f"[{index}/{len(strip_paths)}] {strip_path.name} "
            f"rows={len(rows)} failures={len(failures)}",
            flush=True,
        )

    exact = sum(1 for row in rows if row.get("exact_mask"))
    shape_ok = [row for row in rows if row.get("shape_equal")]
    summary = {
        "strips_compared": len(rows),
        "failures": failures,
        "local_config": config_info,
        "color_mode": args.color_mode,
        "segmenter_url": args.segmenter_url,
        "exact_masks": exact,
        "exact_rate": exact / len(rows) if rows else 0.0,
        "mean_mask_iou": (
            float(np.mean([row["mask_iou"] for row in shape_ok])) if shape_ok else None
        ),
        "min_mask_iou": (
            float(min(row["mask_iou"] for row in shape_ok)) if shape_ok else None
        ),
        "mean_different_pixels": (
            float(np.mean([row["different_pixels"] for row in shape_ok]))
            if shape_ok
            else None
        ),
        "detection_count_matches": sum(
            1 for row in rows if row.get("detections_count_equal")
        ),
        "serving_is_exact": bool(rows) and exact == len(rows) and not failures,
    }

    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output_dir / "raw_vs_torchserve_per_strip.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["serving_is_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

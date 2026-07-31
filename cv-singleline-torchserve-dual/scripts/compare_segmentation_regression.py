#!/usr/bin/env python3
"""Compare a segmenter endpoint with saved per-strip JSON and PNG outputs."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def result_from_response(payload: dict) -> dict:
    return payload["predictions"][0]["segmentation"]["seg_results"][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--strips", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()

    failed = False
    strips = sorted(args.strips.glob("strip_*.jpg"))
    if not strips:
        raise FileNotFoundError(f"no strip_*.jpg files in {args.strips}")

    for strip_path in strips:
        index = int(strip_path.stem.rsplit("_", 1)[1])
        encoded = base64.b64encode(strip_path.read_bytes()).decode("ascii")
        response = requests.post(
            args.endpoint,
            json={
                "instances": [
                    {
                        "model_name": "segmentation",
                        "strip_id": index,
                        "file": encoded,
                    }
                ]
            },
            timeout=180,
        )
        response.raise_for_status()
        current = result_from_response(response.json())

        baseline_json = args.baseline / f"strip_{index}_mask.json"
        previous = result_from_response(json.loads(baseline_json.read_text()))
        previous_mask = np.array(Image.open(args.baseline / f"strip_{index}_mask.png"))
        current_mask = np.array(
            Image.open(io.BytesIO(base64.b64decode(current["mask"]["img"])))
        )

        old_detections = previous["detections"]
        new_detections = current["detections"]
        classes_equal = [item["class_id"] for item in old_detections] == [
            item["class_id"] for item in new_detections
        ]
        bbox_delta = max(
            (
                abs(old_value - new_value)
                for old_item, new_item in zip(old_detections, new_detections)
                for old_value, new_value in zip(old_item["bbox"], new_item["bbox"])
            ),
            default=0.0,
        )
        confidence_delta = max(
            (
                abs(old_item["confidence"] - new_item["confidence"])
                for old_item, new_item in zip(old_detections, new_detections)
            ),
            default=0.0,
        )
        mask_equal = np.array_equal(previous_mask, current_mask)
        different_pixels = int(np.count_nonzero(previous_mask != current_mask))

        passed = (
            len(old_detections) == len(new_detections)
            and classes_equal
            and bbox_delta <= 1.0
            and confidence_delta <= 0.001
            and mask_equal
        )
        failed |= not passed
        print(
            f"strip_{index}: status={response.status_code} "
            f"counts={len(old_detections)}/{len(new_detections)} "
            f"classes_equal={classes_equal} max_bbox_delta={bbox_delta:.1f} "
            f"max_conf_delta={confidence_delta:.9f} "
            f"mask_equal={mask_equal} diff_pixels={different_pixels}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Hit dual TorchServe segmentation endpoint with strip crop(s).

Shelf full images (test_image.jpg) are for *detection*.
Segmentation expects strip crops (narrow shelf-row images).

Usage (from this directory on the GPU):
  cd /data/vaibhav.singh/SingleLine_deployment/cv-singleline-torchserve-dual/segmentation
  python3 test_torchserve.py
  # or pass multiple strips:
  python3 test_torchserve.py ../test-fixtures/segmentation-comparison/strips/*.jpg
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import requests

DUAL_ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get("TORCHSERVE_URL", "http://127.0.0.1:9000/predictions/segmenter")
DEFAULT_IMAGES = [str(DUAL_ROOT / "test-fixtures" / "segmentation" / "strip_0.jpg")]


def main(image_paths: list[str]) -> None:
    instances = []
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        _, buffer = cv2.imencode(".png", image)
        image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        instances.append(
            {
                "model_name": "segmentation",
                "strip_id": idx,
                "file": image_b64,
            }
        )

    payload = {"instances": instances}
    print(f"Images  : {image_paths}")
    print(f"Sending : POST {URL}")
    response = requests.post(URL, json=payload, timeout=180)

    print("Status Code:", response.status_code)
    try:
        result = response.json()
        print(json.dumps(result, indent=2)[:10000])
    except Exception:
        print(response.text)


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_IMAGES
    main(paths)

"""Hit dual TorchServe detection endpoint with a shelf image.

Usage (from this directory on the GPU):
  cd /data/vaibhav.singh/SingleLine_deployment/cv-singleline-torchserve-dual/detection
  python3 test_torchserve.py
  python3 test_torchserve.py ../test-fixtures/detection/test_img_new.jpg
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests

DUAL_ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get("TORCHSERVE_URL", "http://127.0.0.1:9000/predictions/detector")
DEFAULT_IMAGE = str(DUAL_ROOT / "test-fixtures" / "detection" / "test_image.jpg")


def run_detection(image_path: str = DEFAULT_IMAGE) -> dict:
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "instances": [
            {
                "model_name": "detection",
                "file": b64_image,
            }
        ]
    }

    print(f"Image   : {image_path}")
    print(f"Sending : POST {URL}")
    response = requests.post(URL, json=payload, timeout=180)

    print(f"Status  : {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, indent=2)[:8000]}")
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    run_detection(path)

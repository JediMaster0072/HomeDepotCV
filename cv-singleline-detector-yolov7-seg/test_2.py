"""
test_torchserve.py
------------------
Test both detection and segmentation endpoints against a running TorchServe instance.

Usage:
    python test_torchserve.py --image shelf.jpg               # detection only
    python test_torchserve.py --image shelf.jpg --mode both   # detection + segmentation
    python test_torchserve.py --strips s0.jpg s1.jpg s2.jpg   # segmentation only
"""

import argparse
import base64
import json
import sys
import time

import requests

URL = "http://localhost:8080/predictions/yolov7"


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pretty(response: requests.Response) -> None:
    print(f"Status : {response.status_code}")
    try:
        print("Response:")
        print(json.dumps(response.json(), indent=2, default=str))
    except Exception:
        print("Raw:", response.text)


# ── Test functions ─────────────────────────────────────────────────────────────

def test_ping() -> bool:
    """Check TorchServe is up before sending any inference request."""
    try:
        r = requests.get("http://localhost:8080/ping", timeout=5)
        healthy = r.status_code == 200 and r.json().get("status") == "Healthy"
        print(f"[ping] {'OK — TorchServe is healthy' if healthy else 'FAILED — ' + r.text}")
        return healthy
    except requests.exceptions.ConnectionError:
        print("[ping] FAILED — cannot connect to localhost:8080. Is TorchServe running?")
        return False


def test_model_status() -> None:
    """Print worker status for the yolov7 model."""
    try:
        r = requests.get("http://localhost:8081/models/yolov7", timeout=5)
        print("[model status]")
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f"[model status] skipped — management API not reachable ({e})")


def test_detection(image_path: str) -> None:
    """Send a full shelf image for YOLOv7 label detection."""
    print(f"\n{'='*60}")
    print(f"[detection] image={image_path}")
    print("="*60)

    payload = {
        "instances": [{
            "model_name": "detection",
            "file": encode_image(image_path),
        }]
    }

    t0 = time.time()
    response = requests.post(URL, json=payload, timeout=60)
    elapsed  = int((time.time() - t0) * 1000)
    print(f"Round-trip: {elapsed} ms")
    pretty(response)

    # Summary
    if response.status_code == 200:
        preds      = response.json().get("predictions", [{}])[0]
        detections = preds.get("detections", [])
        records    = preds.get("label_records", [])
        print(f"\n[detection] {len(detections)} bboxes  |  {len(records)} label records")


def test_segmentation(strip_paths: list) -> None:
    """Send pre-cropped strip images for YOLOv7-seg segmentation."""
    print(f"\n{'='*60}")
    print(f"[segmentation] strips={strip_paths}")
    print("="*60)

    instances = [
        {
            "model_name": "segmentation",
            "strip_id": i,
            "file": encode_image(path),
        }
        for i, path in enumerate(strip_paths)
    ]
    payload = {"instances": instances}

    t0 = time.time()
    response = requests.post(URL, json=payload, timeout=120)
    elapsed  = int((time.time() - t0) * 1000)
    print(f"Round-trip: {elapsed} ms")
    pretty(response)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Test TorchServe YOLOv7 handler")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to full shelf image (detection mode)")
    parser.add_argument("--strips", type=str, nargs="+", default=None,
                        help="Paths to strip images (segmentation mode)")
    parser.add_argument("--mode",   type=str,
                        choices=["detection", "segmentation", "both"],
                        default="segmentation",
                        help="Which mode to test (default: detection)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Health check first — abort if TorchServe is not up
    if not test_ping():
        sys.exit(1)

    # 2. Model worker status
    test_model_status()

    # 3. Run requested tests
    mode = args.mode

    if mode in ("detection", "both"):
        if not args.image:
            print("[detection] --image is required for detection mode")
            sys.exit(1)
        test_detection(args.image)

    if mode in ("segmentation", "both"):
        if not args.strips:
            print("[segmentation] --strips is required for segmentation mode")
            sys.exit(1)
        test_segmentation(args.strips)

#!/usr/bin/env python3
"""Health + optional inference smoke tests for the two TorchServe detector containers."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 120) -> object:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ping(url: str, label: str) -> None:
    body = http_json("GET", f"{url}/ping", timeout=10)
    if body.get("status") != "Healthy":
        raise RuntimeError(f"{label} ping not healthy: {body}")
    print(f"[ok] {label} ping")


def model_ready(mgmt_url: str, label: str) -> None:
    body = http_json("GET", f"{mgmt_url}/models/yolov7", timeout=15)
    try:
        workers = body[0].get("workers", [])
    except Exception:
        raise RuntimeError(f"unexpected mgmt response: {body}")
    ready = [w for w in workers if w.get("status") == "READY"]
    if not ready:
        raise RuntimeError(f"{label} has no READY workers: {json.dumps(body, indent=2)}")
    print(f"[ok] {label} model workers READY ({len(ready)}/{len(workers)})")


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_detection(infer_url: str, image_path: Path) -> None:
    payload = {
        "instances": [
            {
                "model_name": "detection",
                "file": encode_image(image_path),
            }
        ]
    }
    data = http_json("POST", f"{infer_url}/predictions/yolov7", payload)
    preds = data.get("predictions", [{}])[0]
    dets = preds.get("detections", [])
    print(f"[ok] detection inference — {len(dets)} bbox(es)")
    if dets:
        print(f"     first detection: {dets[0]}")


def test_segmentation(infer_url: str, strip_path: Path) -> None:
    payload = {
        "instances": [
            {
                "model_name": "segmentation",
                "strip_id": 0,
                "file": encode_image(strip_path),
            }
        ]
    }
    data = http_json("POST", f"{infer_url}/predictions/yolov7", payload)
    seg_node = data.get("predictions", [{}])[0].get("segmentation")
    if seg_node is None:
        raise RuntimeError(f"segmentation response missing 'segmentation' key: {data}")
    print("[ok] segmentation inference — response contains segmentation payload")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--det-url", default="http://127.0.0.1:9000")
    p.add_argument("--seg-url", default="http://127.0.0.1:10000")
    p.add_argument("--det-image", type=Path, default=None, help="Optional shelf image for detection infer")
    p.add_argument("--seg-strip", type=Path, default=None, help="Optional strip image for segmentation infer")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    det_infer = args.det_url.rstrip("/")
    seg_infer = args.seg_url.rstrip("/")
    det_mgmt = det_infer.replace(":8080", ":8081").replace(":9000", ":9001")
    seg_mgmt = seg_infer.replace(":8080", ":8081").replace(":10000", ":10001")

    try:
        ping(det_infer, "detection")
        ping(seg_infer, "segmentation")
        model_ready(det_mgmt, "detection")
        model_ready(seg_mgmt, "segmentation")

        if args.det_image:
            if not args.det_image.is_file():
                print(f"[skip] detection image not found: {args.det_image}", file=sys.stderr)
            else:
                test_detection(det_infer, args.det_image)

        if args.seg_strip:
            if not args.seg_strip.is_file():
                print(f"[skip] segmentation strip not found: {args.seg_strip}", file=sys.stderr)
            else:
                test_segmentation(seg_infer, args.seg_strip)

        if not args.det_image and not args.seg_strip:
            print("[info] ping/model checks only — pass --det-image / --seg-strip for inference smoke")
    except (urllib.error.URLError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

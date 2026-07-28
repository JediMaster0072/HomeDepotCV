#!/usr/bin/env python3
"""Health + optional inference smoke tests for TorchServe detectors.

Supports:
  - Dual Option 1 (one container): --base-url http://127.0.0.1:9000
    models: detector + segmenter
  - Legacy two containers: --det-url / --seg-url (models named yolov7)
"""

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


def model_ready(mgmt_url: str, model_name: str, label: str) -> None:
    body = http_json("GET", f"{mgmt_url}/models/{model_name}", timeout=15)
    try:
        workers = body[0].get("workers", [])
    except Exception:
        raise RuntimeError(f"unexpected mgmt response: {body}")
    ready = [w for w in workers if w.get("status") == "READY"]
    if not ready:
        raise RuntimeError(f"{label} has no READY workers: {json.dumps(body, indent=2)}")
    print(f"[ok] {label} model={model_name} workers READY ({len(ready)}/{len(workers)})")


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_detection(infer_url: str, model_name: str, image_path: Path) -> None:
    payload = {
        "instances": [
            {
                "model_name": "detection",
                "file": encode_image(image_path),
            }
        ]
    }
    data = http_json("POST", f"{infer_url}/predictions/{model_name}", payload)
    # TorchServe may return list or dict depending on handler wrapping
    if isinstance(data, list):
        data = data[0] if data else {}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"detection error from server: {data['error']}")
    preds = data.get("predictions", [{}])
    if isinstance(preds, list):
        preds = preds[0] if preds else {}
    if isinstance(preds, dict) and preds.get("error"):
        raise RuntimeError(f"detection error from server: {preds['error']}")
    dets = preds.get("detections", [])
    print(f"[ok] detection inference ({model_name}) — {len(dets)} bbox(es)")
    if dets:
        print(f"     first detection: {dets[0]}")


def test_segmentation(infer_url: str, model_name: str, strip_path: Path) -> None:
    payload = {
        "instances": [
            {
                "model_name": "segmentation",
                "strip_id": 0,
                "file": encode_image(strip_path),
            }
        ]
    }
    data = http_json("POST", f"{infer_url}/predictions/{model_name}", payload)
    if isinstance(data, list):
        data = data[0] if data else {}
    preds = data.get("predictions", [{}])
    if isinstance(preds, list):
        preds = preds[0] if preds else {}
    seg_node = preds.get("segmentation")
    if seg_node is None:
        raise RuntimeError(f"segmentation response missing 'segmentation' key: {data}")
    print(f"[ok] segmentation inference ({model_name}) — response contains segmentation payload")


def infer_mgmt(infer_url: str) -> str:
    """Map common inference ports to management ports."""
    replacements = (
        (":9000", ":9001"),
        (":10000", ":10001"),
        (":8080", ":8081"),
    )
    out = infer_url
    for a, b in replacements:
        if a in out:
            return out.replace(a, b)
    # Fallback: assume mgmt is infer+1 on last port component
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default=None,
        help="Dual Option 1: single TorchServe base (e.g. http://127.0.0.1:9000)",
    )
    p.add_argument("--det-url", default="http://127.0.0.1:9000")
    p.add_argument("--seg-url", default="http://127.0.0.1:10000")
    p.add_argument("--det-model", default=None, help="Override detection model name")
    p.add_argument("--seg-model", default=None, help="Override segmentation model name")
    p.add_argument("--det-image", type=Path, default=None, help="Optional shelf image for detection infer")
    p.add_argument("--seg-strip", type=Path, default=None, help="Optional strip image for segmentation infer")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.base_url:
        base = args.base_url.rstrip("/")
        det_infer = base
        seg_infer = base
        det_model = args.det_model or "detector"
        seg_model = args.seg_model or "segmenter"
        det_mgmt = infer_mgmt(det_infer)
        seg_mgmt = det_mgmt
        mode = "dual"
    else:
        det_infer = args.det_url.rstrip("/")
        seg_infer = args.seg_url.rstrip("/")
        det_model = args.det_model or "yolov7"
        seg_model = args.seg_model or "yolov7"
        det_mgmt = infer_mgmt(det_infer)
        seg_mgmt = infer_mgmt(seg_infer)
        mode = "split"

    try:
        if mode == "dual":
            ping(det_infer, "dual-torchserve")
            model_ready(det_mgmt, det_model, "detection")
            model_ready(seg_mgmt, seg_model, "segmentation")
        else:
            ping(det_infer, "detection")
            ping(seg_infer, "segmentation")
            model_ready(det_mgmt, det_model, "detection")
            model_ready(seg_mgmt, seg_model, "segmentation")

        if args.det_image:
            if not args.det_image.is_file():
                print(f"[skip] detection image not found: {args.det_image}", file=sys.stderr)
            else:
                test_detection(det_infer, det_model, args.det_image)

        if args.seg_strip:
            if not args.seg_strip.is_file():
                print(f"[skip] segmentation strip not found: {args.seg_strip}", file=sys.stderr)
            else:
                test_segmentation(seg_infer, seg_model, args.seg_strip)

        if not args.det_image and not args.seg_strip:
            print("[info] ping/model checks only — pass --det-image / --seg-strip for inference smoke")
    except (urllib.error.URLError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

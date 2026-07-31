#!/usr/bin/env python3
"""Diagnose sources of local vs TorchServe segmentation disagreements."""

from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def fetch_mask(endpoint: str, strip_path: Path, strip_id: int) -> tuple[np.ndarray, int]:
    payload = {
        "instances": [
            {
                "model_name": "segmentation",
                "strip_id": strip_id,
                "file": base64.b64encode(strip_path.read_bytes()).decode("ascii"),
            }
        ]
    }
    response = requests.post(endpoint, json=payload, timeout=180)
    response.raise_for_status()
    result = response.json()["predictions"][0]["segmentation"]["seg_results"][0]
    mask_payload = result.get("mask")
    if not mask_payload or not mask_payload.get("img"):
        return np.zeros((1, 1), dtype=bool), 0
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(mask_payload["img"])))) > 127
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr, len(result.get("detections") or [])


def jpeg_roundtrip(mask: np.ndarray, quality: int = 95) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buf, format="JPEG", quality=quality)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("L")) > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 1.0


def analyze(
    local_root: Path,
    endpoint: str,
    row: dict[str, str],
    label: str,
) -> None:
    stem = Path(row["image"]).stem
    idx = int(row["strip_index"])
    local_dir = local_root / stem
    strip = local_dir / "strips" / f"strip_{idx}.jpg"
    local_mask_path = local_dir / "strips_viz_orig_mask" / f"{idx}_pred_mask.jpg"
    local_gray = np.array(Image.open(local_mask_path).convert("L"))
    local = local_gray > 127
    served, n_det = fetch_mask(endpoint, strip, idx)
    served_jpeg = jpeg_roundtrip(served)

    only_local = int(np.count_nonzero(local & ~served))
    only_served = int(np.count_nonzero(served & ~local))
    print(f"\n[{label}] {row['image']} strip={idx} csv_iou={row['mask_iou']}")
    print(f"  shapes local/ts={local.shape}/{served.shape} dets={n_det}")
    print(
        f"  white local/ts={int(local.sum())}/{int(served.sum())} "
        f"only_local={only_local} only_ts={only_served}"
    )
    print(
        f"  live_iou={iou(local, served):.4f} "
        f"after_jpeg_roundtrip_of_ts={iou(local, served_jpeg):.4f}"
    )
    print(
        f"  local_unique_levels={len(np.unique(local_gray))} "
        f"min/max={int(local_gray.min())}/{int(local_gray.max())}"
    )


def main() -> int:
    base = Path("/data/vaibhav.singh/SingleLine_deployment")
    local_root = base / "local_deployed_instance_outputs"
    rows = list(
        csv.DictReader(
            (base / "local_vs_torchserve_iou_fair_dockerfile/segmentation_per_strip.csv").open()
        )
    )
    valid = [row for row in rows if row.get("shape_equal") == "True"]
    print(f"total_strips={len(valid)}")
    print(f"exact={sum(int(row['different_pixels']) == 0 for row in valid)}")
    print(f"iou_zero={sum(float(row['mask_iou']) == 0 for row in valid)}")
    print(
        "local_empty_ts_not="
        f"{sum(int(row['local_white_pixels']) == 0 and int(row['torchserve_white_pixels']) > 0 for row in valid)}"
    )
    print(
        "ts_empty_local_not="
        f"{sum(int(row['torchserve_white_pixels']) == 0 and int(row['local_white_pixels']) > 0 for row in valid)}"
    )

    jpeg_like = binary_like = 0
    for row in valid[:100]:
        path = (
            local_root
            / Path(row["image"]).stem
            / "strips_viz_orig_mask"
            / f"{row['strip_index']}_pred_mask.jpg"
        )
        uniq = np.unique(np.array(Image.open(path).convert("L")))
        if len(uniq) <= 3 and set(uniq.tolist()).issubset({0, 1, 254, 255}):
            binary_like += 1
        elif len(uniq) > 10:
            jpeg_like += 1
    print(f"sample100 binary_like={binary_like} jpeg_like_many_levels={jpeg_like}")

    endpoint = "http://127.0.0.1:12000/predictions/segmenter"
    nonempty = [
        row
        for row in sorted(valid, key=lambda item: float(item["mask_iou"]))
        if int(row["local_white_pixels"]) > 0 or int(row["torchserve_white_pixels"]) > 0
    ]
    analyze(local_root, endpoint, nonempty[0], "worst")
    analyze(local_root, endpoint, nonempty[len(nonempty) // 2], "median")
    analyze(
        local_root,
        endpoint,
        sorted(valid, key=lambda item: -float(item["mask_iou"]))[0],
        "best",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

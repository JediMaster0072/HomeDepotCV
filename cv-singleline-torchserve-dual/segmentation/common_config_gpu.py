"""Configuration and import isolation for the segmentation-only package."""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Optional, Tuple

_PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
_YOLO_MODULE_PREFIXES: Tuple[str, ...] = ("models", "utils", "segment")


def _prepend_path(path: str) -> None:
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def purge_yolo_modules(extra_prefixes: Optional[Iterable[str]] = None) -> int:
    prefixes = _YOLO_MODULE_PREFIXES + tuple(extra_prefixes or ())
    removed = 0
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]
            removed += 1
    return removed


def activate_seg_repo(config: Dict) -> str:
    purge_yolo_modules()
    root = config["seg_repo_root"]
    _prepend_path(root)
    return root


def ensure_gpu_only_import_paths(model_dir: Optional[str] = None) -> None:
    _prepend_path(os.path.abspath(model_dir) if model_dir else _PACKAGE_ROOT)


def build_gpu_config(model_dir: Optional[str] = None) -> Dict:
    root = os.path.abspath(model_dir) if model_dir else _PACKAGE_ROOT
    ensure_gpu_only_import_paths(root)
    from service_pipeline_gpu.pipeline_config import CONFIG

    config = CONFIG.copy()
    config.update(
        {
            "seg_repo_root": os.path.join(root, "yolov7-seg"),
            "segmentation_weights": os.path.join(root, "segmentation.pt"),
            "seg_imgsz": 608,
            "seg_conf_thres": 0.25,
            "seg_iou_thres": 0.70,
            "seg_max_det": 300,
            "device": os.environ.get("GPU_PIPELINE_DEVICE", "gpu"),
        }
    )
    return config

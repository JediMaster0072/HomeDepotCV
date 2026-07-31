"""Segmentation-worker config — packaged only inside segmenter.mar."""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Optional, Tuple

_PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
_YOLO_MODULE_PREFIXES: Tuple[str, ...] = ("models", "utils", "segment")


def _remove_path(path: str) -> None:
    if not path:
        return
    while path in sys.path:
        sys.path.remove(path)


def _prepend_path(path: str) -> None:
    if not path:
        return
    _remove_path(path)
    sys.path.insert(0, path)


def purge_yolo_modules(extra_prefixes: Optional[Iterable[str]] = None) -> int:
    prefixes = tuple(_YOLO_MODULE_PREFIXES)
    if extra_prefixes:
        prefixes = prefixes + tuple(extra_prefixes)
    removed = 0
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]
            removed += 1
    return removed


def activate_seg_repo(config: Dict) -> str:
    purge_yolo_modules()
    seg_root = config.get("seg_repo_root") or os.path.join(_PACKAGE_ROOT, "yolov7-seg")
    _prepend_path(seg_root)
    return seg_root


def activate_det_repo(config: Dict) -> str:
    raise RuntimeError("activate_det_repo called in segmentation-only worker")


def ensure_gpu_only_import_paths(model_dir: Optional[str] = None) -> None:
    """Put MAR unpack root on sys.path so `import service_pipeline_gpu` works.

    Do NOT prepend the service_pipeline_gpu/ directory itself — that breaks the
    package import (Python would look for service_pipeline_gpu/service_pipeline_gpu).
    """
    root = os.path.abspath(model_dir) if model_dir else _PACKAGE_ROOT
    _prepend_path(root)


def build_gpu_config(model_dir: Optional[str] = None) -> Dict:
    root = os.path.abspath(model_dir) if model_dir else _PACKAGE_ROOT
    ensure_gpu_only_import_paths(root)
    from service_pipeline_gpu.pipeline_config import CONFIG  # noqa: E402

    config = CONFIG.copy()
    config["seg_repo_root"] = os.path.join(root, "yolov7-seg")
    config["segmentation_weights"] = os.path.join(root, "segmentation.pt")
    config["seg_imgsz"] = 608
    config["seg_conf_thres"] = 0.25
    config["seg_iou_thres"] = 0.70
    config["seg_max_det"] = 300
    config["det_repo_root"] = ""
    config["detection_weights"] = ""
    config["detection_imgsz"] = 1280
    config["detection_conf_thres"] = 0.15
    config["detection_iou_thres"] = 0.45
    config["detection_max_det"] = 300
    config["device"] = os.environ.get("GPU_PIPELINE_DEVICE", "gpu")
    config["debug"] = False
    config["save_debug_artifacts"] = False
    config["save_result_json"] = False
    config["save_annotated_image"] = False
    return config

import os
import sys
from typing import Dict

ROOT = os.path.abspath(os.path.dirname(__file__))
PIPELINE_GPU_DIR = os.path.join(ROOT, "servicepipelinegpu")
YOLOV7_DET_ROOT = os.path.join(ROOT, "yolov7")
YOLOV7_SEG_ROOT = os.path.join(ROOT, "yolov7-seg")


def _prepend_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def remove_pipeline_paths() -> None:
    for path in (PIPELINE_GPU_DIR,):
        while path in sys.path:
            sys.path.remove(path)


def ensure_gpu_only_import_paths() -> None:
    _prepend_path(PIPELINE_GPU_DIR)
    _prepend_path(YOLOV7_DET_ROOT)
    _prepend_path(YOLOV7_SEG_ROOT)


def build_gpu_config() -> Dict:
    """Build and return the full pipeline config dict for the GPU service.

    Two models only:
      Stage 1 — YOLOv7 label detection   (best.pt)
      Stage 4 — YOLOv7-seg segmentation  (segmentation.pt)
    """
    ensure_gpu_only_import_paths()

    from service_pipeline_gpu.pipeline_config import CONFIG  # noqa: E402

    config = CONFIG.copy()

    # ── detection model (YOLOv7) ──────────────────────────────────────────
    config["det_repo_root"] = YOLOV7_DET_ROOT
    config["detection_weights"] = os.path.join(ROOT, "best.pt")
    config["detection_imgsz"] = 1024
    config["detection_conf_thres"] = 0.25
    config["detection_iou_thres"] = 0.45
    config["detection_max_det"] = 1000

    # ── segmentation model (YOLOv7-seg) ──────────────────────────────────
    config["seg_repo_root"] = YOLOV7_SEG_ROOT
    config["segmentation_weights"] = os.path.join(ROOT, "segmentation.pt")
    config["seg_imgsz"] = 608
    config["seg_conf_thres"] = 0.25
    config["seg_iou_thres"] = 0.7
    config["seg_max_det"] = 300

    # ── runtime flags ──────────────────────────────────────────────────────
    config["device"] = os.environ.get("GPU_PIPELINE_DEVICE", "gpu")
    config["debug"] = False
    config["save_debug_artifacts"] = False
    config["save_result_json"] = False
    config["save_annotated_image"] = False

    return config

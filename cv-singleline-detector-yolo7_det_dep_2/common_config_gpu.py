import os
import sys
from typing import Dict, Iterable, Optional, Tuple

ROOT = os.path.abspath(os.path.dirname(__file__))
PIPELINE_GPU_DIR = os.path.join(ROOT, "service_pipeline_gpu")
YOLOV7_DET_ROOT = os.path.join(ROOT, "yolov7")
YOLOV7_SEG_ROOT = os.path.join(ROOT, "yolov7-seg")

# Module namespaces shared by yolov7/ and yolov7-seg/ that collide in one process.
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
    """Drop cached YOLO repo modules so the next import resolves from sys.path.

    Returns the number of modules removed from sys.modules.
    """
    prefixes = tuple(_YOLO_MODULE_PREFIXES)
    if extra_prefixes:
        prefixes = prefixes + tuple(extra_prefixes)

    removed = 0
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]
            removed += 1
    return removed


def activate_yolo_repo(repo_root: str, *, block_roots: Iterable[str] = ()) -> str:
    """Purge YOLO module cache, hide other repo roots, and prioritize repo_root."""
    purge_yolo_modules()
    for blocked in block_roots:
        _remove_path(blocked)
    if repo_root:
        _prepend_path(repo_root)
    return repo_root


def activate_det_repo(config: Dict) -> str:
    det_root = config.get("det_repo_root", YOLOV7_DET_ROOT)
    seg_root = config.get("seg_repo_root", YOLOV7_SEG_ROOT)
    return activate_yolo_repo(det_root, block_roots=(seg_root,))


def activate_seg_repo(config: Dict) -> str:
    det_root = config.get("det_repo_root", YOLOV7_DET_ROOT)
    seg_root = config.get("seg_repo_root", YOLOV7_SEG_ROOT)
    return activate_yolo_repo(seg_root, block_roots=(det_root,))


def ensure_gpu_only_import_paths() -> None:
    """Expose only in-repo pipeline packages — never both YOLO repos at once."""
    _prepend_path(PIPELINE_GPU_DIR)


def remove_pipeline_paths() -> None:
    for path in (PIPELINE_GPU_DIR,):
        _remove_path(path)


def build_gpu_config() -> Dict:
    """Build and return the full pipeline config dict for the GPU service.

    Two models only:
      Stage 1 — YOLOv7 label detection   (best.pt)
      Stage 2 — YOLOv7-seg segmentation  (segmentation.pt)
    """
    ensure_gpu_only_import_paths()

    from service_pipeline_gpu.pipeline_config import CONFIG  # noqa: E402

    config = CONFIG.copy()

    # ── detection model (YOLOv7) ──────────────────────────────────────────
    config["det_repo_root"] = YOLOV7_DET_ROOT
    config["detection_weights"] = os.path.join(ROOT, "best.pt")
    config["detection_imgsz"] = 1280
    config["detection_conf_thres"] = 0.15
    config["detection_iou_thres"] = 0.45
    config["detection_max_det"] = 300

    # ── segmentation model (YOLOv7-seg) ──────────────────────────────────
    config["seg_repo_root"] = YOLOV7_SEG_ROOT
    config["segmentation_weights"] = os.path.join(ROOT, "segmentation.pt")
    config["seg_imgsz"] = 640
    config["seg_conf_thres"] = 0.25
    config["seg_iou_thres"] = 0.45
    config["seg_max_det"] = 1000

    # ── runtime flags ──────────────────────────────────────────────────────
    config["device"] = os.environ.get("GPU_PIPELINE_DEVICE", "gpu")
    config["debug"] = False
    config["save_debug_artifacts"] = False
    config["save_result_json"] = False
    config["save_annotated_image"] = False

    return config

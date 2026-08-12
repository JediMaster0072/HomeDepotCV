"""
SKU Reading Pipeline — Stage 1: Label Detection
================================================
Runs YOLOv7 inference on a full shelf image.

Input:  RGB ndarray (full shelf image)
Output: list[dict] raw detections with bbox / confidence / class_id
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from common_config_gpu import activate_det_repo

logger = logging.getLogger("sku_pipeline.stage1")


class Stage1Detection:
    """YOLOv7 label detection on a full shelf image."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.names = None
        self.stride = 32
        self._half = False
        self._imgsz = config.get("detection_imgsz", 640)
        self._conf = config.get("detection_conf_thres", 0.25)
        self._iou = config.get("detection_iou_thres", 0.45)
        self._device = torch.device(
            "cuda"
            if config.get("device", "gpu") == "gpu" and torch.cuda.is_available()
            else "cpu"
        )
        self._letterbox = None
        self._non_max_suppression = None
        self._scale_coords = None
        self._time_synchronized = None

    def _bind_detector_utils(self) -> None:
        activate_det_repo(self.config)
        from utils.datasets import letterbox
        from utils.general import non_max_suppression, scale_coords
        from utils.torch_utils import time_synchronized

        self._letterbox = letterbox
        self._non_max_suppression = non_max_suppression
        self._scale_coords = scale_coords
        self._time_synchronized = time_synchronized

    def load_model(self) -> None:
        """Load YOLOv7 detection weights and warm up the GPU kernels."""
        activate_det_repo(self.config)
        from models.experimental import attempt_load
        from utils.general import check_img_size, set_logging
        from utils.torch_utils import TracedModel

        set_logging()
        weights = self.config["detection_weights"]
        model = attempt_load(weights, map_location=self._device)
        self.stride = int(model.stride.max())
        self._imgsz = check_img_size(self._imgsz, s=self.stride)
        model = TracedModel(model, self._device, self._imgsz)

        self._half = self._device.type != "cpu"
        if self._half:
            model.half()

        model.eval()
        for module in model.modules():
            module.training = False

        if self._device.type != "cpu":
            cudnn.benchmark = True
            dummy = torch.zeros(1, 3, self._imgsz, self._imgsz).to(self._device)
            if self._half:
                dummy = dummy.half()
            model(dummy)

        self.model = model
        self.names = model.module.names if hasattr(model, "module") else model.names
        self._bind_detector_utils()
        logger.info(
            "[Stage1] model loaded | device=%s | imgsz=%s | half=%s | classes=%s",
            self._device,
            self._imgsz,
            self._half,
            self.names,
        )

    def run_inference(self, image_rgb: np.ndarray) -> List[Dict]:
        """
        Letterbox → tensor → YOLOv7 → NMS.

        Expects an RGB image (no BGR↔RGB channel flip in this stage).
        Returns a flat list of raw detection dicts.
        """
        if self._letterbox is None:
            self._bind_detector_utils()

        letterbox = self._letterbox
        non_max_suppression = self._non_max_suppression
        scale_coords = self._scale_coords
        time_synchronized = self._time_synchronized

        h0, w0 = image_rgb.shape[:2]
        img, _, _ = letterbox(image_rgb, self._imgsz, stride=self.stride)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self._device)
        img = img.half() if self._half else img.float()
        img = img / 255.0

        t1 = time_synchronized()
        with torch.no_grad():
            raw = self.model(img)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(
            raw, self._conf, self._iou, classes=None, agnostic=False
        )
        t3 = time_synchronized()
        logger.info(
            "[Stage1] inference %.1fms  NMS %.1fms",
            (t2 - t1) * 1e3,
            (t3 - t2) * 1e3,
        )

        detections: List[Dict] = []
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (h0, w0)).round()
            for *xyxy, conf, cls in det.cpu().numpy():
                detections.append(
                    {
                        "bbox": [float(x) for x in xyxy],
                        "confidence": float(conf),
                        "class_id": int(cls),
                    }
                )
        return detections

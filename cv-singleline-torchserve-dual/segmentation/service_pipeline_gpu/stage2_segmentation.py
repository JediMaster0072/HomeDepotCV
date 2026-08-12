"""
SKU Reading Pipeline — Stage 2: Segmentation
=============================================
Runs YOLOv7-seg on shelf-strip crops and returns a combined binary mask.

Input:  list of StripInfo (each with an RGB strip image)
Output: list[dict] with strip_index, detections, and base64 PNG mask
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from codec_gpu import numpy_image_to_base64_png

logger = logging.getLogger("sku_pipeline.stage2")


class Stage2Segmentation:
    """YOLOv7-seg segmentation on one or more strip crops."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = False
        self.repo_root = config["seg_repo_root"]
        self.weights = config["segmentation_weights"]
        self.model = None
        self.names = None
        self.stride = 32
        self.pt = True
        self.imgsz = config.get("seg_imgsz", 608)
        self.conf_thres = config.get("seg_conf_thres", 0.25)
        self.iou_thres = config.get("seg_iou_thres", 0.70)
        self.max_det = config.get("seg_max_det", 300)
        self._letterbox = None
        self._non_max_suppression = None
        self._scale_coords = None
        self._process_mask_fn = None

    def _ensure_seg_path(self) -> None:
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

    def _bind_seg_utils(self) -> None:
        self._ensure_seg_path()
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_coords
        from utils.segment.general import process_mask

        self._letterbox = letterbox
        self._non_max_suppression = non_max_suppression
        self._scale_coords = scale_coords
        self._process_mask_fn = process_mask

    def load_model(self) -> None:
        """Load YOLOv7-seg weights and warm up the GPU kernels."""
        self._ensure_seg_path()
        from models.common import DetectMultiBackend
        from utils.general import check_img_size

        started = time.time()
        self.model = DetectMultiBackend(
            self.weights,
            device=self.device,
            dnn=False,
            data=None,
            fp16=self.half,
        )
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        self._bind_seg_utils()
        logger.info(
            "[Stage2] model loaded in %d ms device=%s imgsz=%s half=%s",
            int((time.time() - started) * 1000),
            self.device,
            self.imgsz,
            self.half,
        )

    def run_inference(self, strips: Sequence[Any]) -> List[Dict[str, Any]]:
        """Run segmentation on each strip and return per-strip results."""
        results: List[Dict[str, Any]] = []
        for strip in strips:
            single = self._run_single(strip.strip_image)
            results.append(
                {
                    "strip_index": strip.strip_index,
                    "detections": single["detections"],
                    "mask": single["mask"],
                }
            )
        return results

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Letterbox + CHW tensor. Expects RGB (no BGR↔RGB channel flip)."""
        if self._letterbox is None:
            self._bind_seg_utils()

        img = self._letterbox(
            image_rgb, self.imgsz, stride=self.stride, auto=self.pt
        )[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor[None]
        return tensor

    @staticmethod
    def _find_proto(output: Any) -> torch.Tensor:
        candidates: List[torch.Tensor] = []

        def flatten(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    flatten(item)
            elif isinstance(value, torch.Tensor):
                candidates.append(value)

        flatten(output)
        proto = next(
            (tensor for tensor in candidates if tensor.ndim == 4 and tensor.shape[1] == 32),
            None,
        )
        if proto is None:
            shapes = [tensor.shape for tensor in candidates]
            raise ValueError(f"[Stage2] Proto tensor not found. Candidates: {shapes}")
        return proto

    def _process_mask(
        self,
        proto: torch.Tensor,
        mask_coeffs: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        input_hw: Tuple[int, int],
        orig_hw: Tuple[int, int],
    ) -> np.ndarray:
        if self._process_mask_fn is None:
            self._bind_seg_utils()

        masks_tensor = self._process_mask_fn(
            proto[0], mask_coeffs, boxes_xyxy, input_hw, upsample=True
        )
        masks_np = (masks_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255
        orig_h, orig_w = orig_hw
        resized = [
            cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            for mask in masks_np
        ]
        return np.stack(resized, axis=0)

    def _run_single(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        if self._non_max_suppression is None:
            self._bind_seg_utils()

        orig_h, orig_w = image_rgb.shape[:2]
        img_tensor = self._preprocess(image_rgb)
        input_hw = (img_tensor.shape[2], img_tensor.shape[3])

        with torch.no_grad():
            out = self.model(img_tensor, augment=False, visualize=False)

        if not isinstance(out, tuple):
            raise ValueError("[Stage2] Model output must be a tuple for segmentation")

        pred_raw = out[0]
        proto = self._find_proto(out[1])
        pred = self._non_max_suppression(
            pred_raw,
            self.conf_thres,
            self.iou_thres,
            classes=None,
            agnostic=False,
            max_det=self.max_det,
            nm=32,
        )[0]

        if pred is None or len(pred) == 0:
            return {"detections": [], "mask": None}

        boxes_xyxy = pred[:, :4]
        mask_coeffs = pred[:, 6:]
        masks_np = self._process_mask(
            proto, mask_coeffs, boxes_xyxy, input_hw, (orig_h, orig_w)
        )
        pred[:, :4] = self._scale_coords(
            input_hw, pred[:, :4], (orig_h, orig_w)
        ).round()

        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        detections: List[Dict[str, Any]] = []
        for index, det in enumerate(pred.cpu()):
            x1, y1, x2, y2, conf, cls_id = det[:6].tolist()
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                }
            )
            binary_mask = cv2.bitwise_or(binary_mask, masks_np[index])

        mask_dict: Optional[Dict[str, Any]] = {
            "img": numpy_image_to_base64_png(binary_mask),
            "height": orig_h,
            "width": orig_w,
        }
        logger.debug(
            "[Stage2] %d detections mask=%dx%d", len(detections), orig_h, orig_w
        )
        return {"detections": detections, "mask": mask_dict}

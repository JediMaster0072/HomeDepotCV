"""
SKU Reading Pipeline — Stage 1: Label Detection
================================================
Runs YOLOv7 inference on a base (shelf) image to detect labels.
Produces LabelRecords sorted left-to-right for downstream stages.

Input:  BGR ndarray (full shelf image)
Output: list[LabelRecord], list[dict] raw detections
"""

import sys
import time
import logging
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .label_record import LabelRecord
from .pipeline_config import (
    DETECTION_CLASSES,
    DETECTION_CLASS_LABEL,
    DETECTION_SUB_ELEMENT_IDS,
)

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
            "cuda" if config.get("device", "gpu") == "gpu" and torch.cuda.is_available()
            else "cpu"
        )

    def load_model(self) -> None:
        """Load YOLOv7 detection weights."""
        det_root = self.config.get("det_repo_root", "")
        seg_root = self.config.get("seg_repo_root", "")

        removed = False
        if seg_root and seg_root in sys.path:
            sys.path.remove(seg_root)
            removed = True

        if det_root and det_root not in sys.path:
            sys.path.insert(0, det_root)

        try:
            from models.experimental import attempt_load
            from utils.general import check_img_size, set_logging
            from utils.torch_utils import TracedModel
        finally:
            if removed and seg_root not in sys.path:
                sys.path.append(seg_root)

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
        for m in model.modules():
            m.training = False

        if self._device.type != "cpu":
            cudnn.benchmark = True
            dummy = torch.zeros(1, 3, self._imgsz, self._imgsz).to(self._device)
            if self._half:
                dummy = dummy.half()
            model(dummy)

        self.model = model
        self.names = model.module.names if hasattr(model, "module") else model.names
        logger.info(
            f"[Stage1] model loaded | device={self._device} "
            f"| imgsz={self._imgsz} | half={self._half} | classes={self.names}"
        )

    def run_inference(self, image_bgr: np.ndarray) -> List[Dict]:
        """
        GPU phase: letterbox → tensor → YOLOv7 → NMS.
        Returns a flat list of raw detection dicts.
        """
        det_root = self.config.get("det_repo_root", "")
        seg_root = self.config.get("seg_repo_root", "")

        removed = False
        if seg_root and seg_root in sys.path:
            sys.path.remove(seg_root)
            removed = True

        if det_root and det_root not in sys.path:
            sys.path.insert(0, det_root)

        try:
            from utils.general import non_max_suppression, scale_coords
            from utils.torch_utils import time_synchronized
            from utils.datasets import letterbox
        finally:
            if removed and seg_root not in sys.path:
                sys.path.append(seg_root)

        h0, w0 = image_bgr.shape[:2]

        img, ratio, (dw, dh) = letterbox(image_bgr, self._imgsz, stride=self.stride)
        img = img[:, :, ::-1].copy()
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
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
            f"[Stage1] inference {(t2-t1)*1e3:.1f}ms  NMS {(t3-t2)*1e3:.1f}ms"
        )

        detections: List[Dict] = []
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (h0, w0)).round()
            for *xyxy, conf, cls in det.cpu().numpy():
                detections.append({
                    "bbox": [float(x) for x in xyxy],
                    "confidence": float(conf),
                    "class_id": int(cls),
                })

        return detections

    def build_label_records(
        self,
        raw_detections: List[Dict],
        source_image: str = "in-memory",
    ) -> List[LabelRecord]:
        """
        CPU phase: separate labels / sub-elements, associate children,
        return LabelRecords sorted left-to-right.
        """
        t0 = time.time()

        labels = [d for d in raw_detections if d["class_id"] == DETECTION_CLASS_LABEL]
        sub_elements = [d for d in raw_detections if d["class_id"] in DETECTION_SUB_ELEMENT_IDS]

        label_infos = self._associate_children(labels, sub_elements, source_image)
        records = self._create_records(label_infos, source_image)

        with_ch = sum(1 for r in records if r.has_children)
        without_ch = sum(1 for r in records if not r.has_children)
        logger.info(
            f"[Stage1] {source_image}: {len(records)} labels "
            f"({with_ch} with children, {without_ch} without) "
            f"[{(time.time()-t0)*1e3:.0f}ms]"
        )
        return records

    def _associate_children(
        self,
        labels: List[Dict],
        sub_elements: List[Dict],
        img_name: str,
    ) -> List[Dict]:
        results = [
            {"label_bbox": lb["bbox"], "label_score": lb["confidence"], "children": []}
            for lb in labels
        ]

        multi_match = unmatched = 0
        for se in sub_elements:
            candidates = [
                idx for idx, lb in enumerate(labels)
                if self._is_child_of(se["bbox"], lb["bbox"])
            ]
            if not candidates:
                unmatched += 1
                continue
            if len(candidates) > 1:
                multi_match += 1
            best = max(
                candidates,
                key=lambda idx: self._child_parent_score(
                    se["bbox"], labels[idx]["bbox"], labels[idx]["confidence"]
                ),
            )
            results[best]["children"].append(se)

        sorted_order = sorted(range(len(labels)), key=lambda i: labels[i]["bbox"][0])
        final_id = {orig: fid for fid, orig in enumerate(sorted_order)}

        for i, r in enumerate(results):
            type_counts: Dict[str, int] = {}
            for c in r["children"]:
                name = DETECTION_CLASSES.get(c["class_id"], "?")
                type_counts[name] = type_counts.get(name, 0) + 1
            for name, cnt in type_counts.items():
                if cnt > 1:
                    logger.warning(
                        f"  [Stage1] Label#{final_id.get(i, i)} has {cnt}x {name} "
                        f"— possible NMS cross-class issue"
                    )

        matched = sum(1 for r in results if r["children"])
        logger.info(
            f"  [Stage1] {img_name}: {len(labels)} labels | "
            f"{matched} with children | "
            f"multi_match={multi_match} unmatched={unmatched}"
        )
        return results

    def _create_records(
        self,
        label_infos: List[Dict],
        img_name: str,
    ) -> List[LabelRecord]:
        label_infos.sort(key=lambda li: li["label_bbox"][0])
        records = []
        for idx, li in enumerate(label_infos):
            has_ch = len(li["children"]) > 0
            records.append(LabelRecord(
                label_id=idx,
                source_image=img_name,
                original_bbox=li["label_bbox"],
                detection_score=li["label_score"],
                has_children=has_ch,
                children=li["children"] if has_ch else None,
                status="detected",
            ))
        return records

    @staticmethod
    def _child_parent_score(
        child_bbox: List[float],
        parent_bbox: List[float],
        parent_score: float,
    ) -> Tuple[float, float, float, float]:
        containment = Stage1Detection._overlap_ratio(child_bbox, parent_bbox)
        ccx = (child_bbox[0] + child_bbox[2]) / 2.0
        ccy = (child_bbox[1] + child_bbox[3]) / 2.0
        pcx = (parent_bbox[0] + parent_bbox[2]) / 2.0
        pcy = (parent_bbox[1] + parent_bbox[3]) / 2.0
        dist2 = (ccx - pcx) ** 2 + (ccy - pcy) ** 2
        parent_area = max(
            (parent_bbox[2] - parent_bbox[0]) * (parent_bbox[3] - parent_bbox[1]), 1e-6
        )
        return (containment, -dist2, -parent_area, parent_score)

    @staticmethod
    def _overlap_ratio(child_bbox: List[float], parent_bbox: List[float]) -> float:
        ix1 = max(child_bbox[0], parent_bbox[0])
        iy1 = max(child_bbox[1], parent_bbox[1])
        ix2 = min(child_bbox[2], parent_bbox[2])
        iy2 = min(child_bbox[3], parent_bbox[3])
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        child_area = max(
            (child_bbox[2] - child_bbox[0]) * (child_bbox[3] - child_bbox[1]), 1e-6
        )
        return ((ix2 - ix1) * (iy2 - iy1)) / child_area

    @staticmethod
    def _is_child_of(
        child_bbox: List[float],
        parent_bbox: List[float],
        threshold: float = 0.5,
    ) -> bool:
        cx = (child_bbox[0] + child_bbox[2]) / 2.0
        cy = (child_bbox[1] + child_bbox[3]) / 2.0
        px1, py1, px2, py2 = parent_bbox
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True
        ix1 = max(child_bbox[0], px1)
        iy1 = max(child_bbox[1], py1)
        ix2 = min(child_bbox[2], px2)
        iy2 = min(child_bbox[3], py2)
        if ix1 >= ix2 or iy1 >= iy2:
            return False
        child_area = (child_bbox[2] - child_bbox[0]) * (child_bbox[3] - child_bbox[1])
        return ((ix2 - ix1) * (iy2 - iy1)) / max(child_area, 1e-6) >= threshold

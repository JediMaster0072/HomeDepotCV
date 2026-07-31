# service_pipeline_gpu/stage2_segmentation.py

import sys
import time
import logging

import cv2
import numpy as np
import torch
from codec_gpu import numpy_image_to_base64_png

logger = logging.getLogger("sku_pipeline.stage2")


class Stage2Segmentation:
    """
    YOLOv7-seg segmentation stage.
    Returns a single combined binary mask per strip.

    Assumes Dockerfile exposes yolov7-seg as:
      - seg_models
      - seg_utils
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.half = self.device.type != "cuda"
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

    def _ensure_seg_path(self):
        seg_root = self.repo_root
        if seg_root not in sys.path:
            sys.path.insert(0, seg_root)

    def load_model(self):
        self._ensure_seg_path()

        from models.common import DetectMultiBackend
        from utils.general import check_img_size

        t0 = time.time()
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=None, fp16=self.half)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

        logger.info(f"[Stage2] model loaded in {int((time.time() - t0) * 1000)} ms device={self.device} imgsz={self.imgsz} half={self.half}")

    def run_inference(self, strips):
        results = []
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

    def _preprocess(self, image_bgr):
        self._ensure_seg_path()
        from utils.augmentations import letterbox

        img = letterbox(image_bgr, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img[:, :, ::-1]
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor[None]
        return tensor

    @staticmethod
    def _find_proto(output):
        candidates = []

        def flatten(x):
            if isinstance(x, (list, tuple)):
                for item in x:
                    flatten(item)
            elif isinstance(x, torch.Tensor):
                candidates.append(x)

        flatten(output)
        proto = next((t for t in candidates if t.ndim == 4 and t.shape[1] == 32), None)
        if proto is None:
            shapes = [c.shape for c in candidates]
            raise ValueError(f"[Stage2] Proto tensor not found. Candidates: {shapes}")
        return proto

    def _process_mask(self, proto, mask_coeffs, boxes_xyxy, input_hw, orig_hw):
        self._ensure_seg_path()
        from utils.segment.general import process_mask

        masks_tensor = process_mask(proto[0], mask_coeffs, boxes_xyxy, input_hw, upsample=True)
        masks_np = masks_tensor.cpu().numpy()
        masks_np = (masks_np > 0.5).astype(np.uint8) * 255

        orig_h, orig_w = orig_hw
        resized = [cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) for m in masks_np]
        return np.stack(resized, axis=0)

    def _run_single(self, image_bgr):
        self._ensure_seg_path()
        from utils.general import non_max_suppression, scale_coords

        orig_h, orig_w = image_bgr.shape[:2]
        img_tensor = self._preprocess(image_bgr)
        input_hw = (img_tensor.shape[2], img_tensor.shape[3])

        with torch.no_grad():
            out = self.model(img_tensor, augment=False, visualize=False)

        if not isinstance(out, tuple):
            raise ValueError("[Stage2] Model output must be a tuple for segmentation")

        pred_raw = out[0]
        proto = self._find_proto(out[1])

        pred = non_max_suppression(pred_raw, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=self.max_det, nm=32)[0]

        if pred is None or len(pred) == 0:
            return {"detections": [], "mask": None}

        boxes_xyxy = pred[:, :4]
        mask_coeffs = pred[:, 6:]

        masks_np = self._process_mask(proto, mask_coeffs, boxes_xyxy, input_hw, (orig_h, orig_w))

        pred[:, :4] = scale_coords(input_hw, pred[:, :4], (orig_h, orig_w)).round()

        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        detections = []
        pred_cpu = pred.cpu()

        for i, det in enumerate(pred_cpu):
            x1, y1, x2, y2, conf, cls_id = det[:6].tolist()
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                }
            )
            binary_mask = cv2.bitwise_or(binary_mask, masks_np[i])

        mask_dict = {
            "img": numpy_image_to_base64_png(binary_mask),
            "height": orig_h,
            "width": orig_w,
        }

        logger.debug(f"[Stage2] {len(detections)} detections mask={orig_h}x{orig_w}")
        return {"detections": detections, "mask": mask_dict}

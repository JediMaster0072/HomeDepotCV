"""TorchServe handler for the standalone YOLOv7 detection model."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from ts.torch_handler.base_handler import BaseHandler


class YoloV7Handler(BaseHandler):
    """Accept full shelf images and return detection bounding boxes."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initialized = False
        self._init_error = None
        self.stage1 = None

    def initialize(self, context):
        model_dir = Path(context.system_properties["model_dir"]).resolve()
        app_home = Path(os.environ.get("APP_HOME", "/app")).resolve()
        source_root = app_home if (app_home / "yolov7").is_dir() else model_dir

        for path in (str(source_root), str(model_dir)):
            while path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)

        try:
            from codec_gpu import base64_png_to_numpy_image
            from common_config_gpu import build_gpu_config
            from service_pipeline_gpu.stage1_detection import Stage1Detection

            self._decode_image = base64_png_to_numpy_image
            config = build_gpu_config(model_dir=str(source_root))
            serialized = context.manifest["model"]["serializedFile"]
            weights_path = model_dir / serialized
            if not weights_path.is_file():
                weights_path = source_root / serialized
            if not weights_path.is_file():
                raise FileNotFoundError(f"detection weights missing: {weights_path}")
            config["detection_weights"] = str(weights_path)

            self.stage1 = Stage1Detection(config)
            started = time.time()
            print(
                f"[DetectionHandler] loading YOLOv7 from {source_root} "
                f"weights={weights_path} cuda={torch.cuda.is_available()}"
            )
            self.stage1.load_model()
            print(f"[DetectionHandler] ready in {int((time.time() - started) * 1000)} ms")
            self.initialized = True
            self._init_error = None
        except Exception as exc:
            self.initialized = False
            self._init_error = str(exc)
            print(f"[DetectionHandler] initialization failed: {exc}")
            traceback.print_exc()

    def handle(self, data, context):
        if not self.initialized:
            return [{"error": self._init_error or "detection model not initialized"}]
        try:
            return self.postprocess(self.inference(self.preprocess(data)))
        except Exception as exc:
            print(f"[DetectionHandler] request failed: {exc}")
            traceback.print_exc()
            return [{"error": str(exc)}]

    def preprocess(self, request):
        body = request[0].get("body") or request[0]
        if isinstance(body, (bytes, bytearray)):
            body = json.loads(body)
        instance = body["instances"][0]
        model_name = instance.get("model_name", "detection")
        if model_name != "detection":
            raise ValueError(f"expected model_name='detection', got {model_name!r}")
        return self._decode_image(instance["file"])

    def inference(self, image_rgb):
        return self.stage1.run_inference(image_rgb)

    @staticmethod
    def postprocess(raw_detections):
        boxes = sorted(
            [
                [
                    detection["bbox"][0],
                    detection["bbox"][1],
                    detection["bbox"][2],
                    detection["bbox"][3],
                    detection["confidence"],
                    detection["class_id"],
                ]
                for detection in raw_detections
            ],
            key=lambda box: (box[1], box[0]),
        )
        return [{"predictions": [{"detections": boxes}]}]

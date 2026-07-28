"""
TorchServe handler — detection only (detector.mar worker process).

Request:
  { "instances": [{ "model_name": "detection", "file": "<base64-image>" }] }

Endpoint: POST /predictions/detector
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import torch
from ts.torch_handler.base_handler import BaseHandler


class DetectionHandler(BaseHandler):
    """Loads only Stage1 / yolov7 from this MAR's model_dir."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initialized = False
        self._init_error = None
        self.config = None
        self.stage1 = None
        self._context = None

    def initialize(self, context):
        self._context = context
        properties = context.system_properties
        model_dir = Path(properties["model_dir"]).resolve()
        serialized = context.manifest["model"]["serializedFile"]
        weights_path = model_dir / serialized

        # Only this MAR unpack dir — never a shared /app YOLO tree.
        unpack = str(model_dir)
        while unpack in sys.path:
            sys.path.remove(unpack)
        sys.path.insert(0, unpack)

        print(
            f"[DetectionHandler] model_dir={model_dir} weights={weights_path} "
            f"exists={weights_path.is_file()} cuda={torch.cuda.is_available()} "
            f"pipeline_pkg={(model_dir / 'service_pipeline_gpu').is_dir()} "
            f"entries={sorted(p.name for p in model_dir.iterdir())[:30]}"
        )

        try:
            from common_config_gpu import build_gpu_config  # noqa: E402
            from codec_gpu import base64_png_to_numpy_image  # noqa: E402
            from service_pipeline_gpu.stage1_detection import Stage1Detection  # noqa: E402

            self._base64_png_to_numpy_image = base64_png_to_numpy_image
            self.config = build_gpu_config(model_dir=str(model_dir))
            if weights_path.is_file():
                self.config["detection_weights"] = str(weights_path)
            else:
                raise FileNotFoundError(f"detection weights missing: {weights_path}")

            self.stage1 = Stage1Detection(self.config)
            t0 = time.time()
            print("[DetectionHandler] Loading Stage 1 — YOLOv7 detection …")
            self.stage1.load_model()
            print(f"[DetectionHandler] Stage 1 ready in {int((time.time() - t0) * 1000)} ms")
            self.initialized = True
            self._init_error = None
        except Exception as e:
            print(f"[DetectionHandler] Initialization failed: {e}")
            traceback.print_exc()
            self.initialized = False
            self._init_error = str(e)

    def handle(self, data, context):
        if not self.initialized:
            # One retry — covers transient CUDA / startup-order failures.
            if self._context is not None:
                print("[DetectionHandler] Not initialized; retrying initialize() …")
                self.initialize(self._context)
            if not self.initialized:
                err = self._init_error or "model not initialized"
                return [{"error": err}]
        try:
            model_input = self.preprocess(data)
            if model_input is None:
                return [{"error": "preprocessing failed"}]
            result = self.inference(model_input)
            return self.postprocess(result)
        except Exception as e:
            print(f"[DetectionHandler] handle() error: {e}")
            traceback.print_exc()
            return [{"error": str(e)}]

    def preprocess(self, request):
        try:
            body = request[0].get("body") or request[0]
            if isinstance(body, (bytes, bytearray)):
                import json

                body = json.loads(body)
            instances = body["instances"]
            instance = instances[0]
            t0 = time.time()
            image_rgb = self._base64_png_to_numpy_image(instance["file"])
            print(
                f"[DetectionHandler] preprocess {int((time.time() - t0) * 1000)} ms "
                f"shape={image_rgb.shape}"
            )
            return {"image_rgb": image_rgb}
        except Exception as e:
            print(f"[DetectionHandler] preprocess error: {e}")
            traceback.print_exc()
            return None

    def inference(self, model_input: dict) -> dict:
        t0 = time.time()
        raw_dets = self.stage1.run_inference(model_input["image_rgb"])
        print(f"[DetectionHandler] inference {int((time.time() - t0) * 1000)} ms")
        return {"raw_dets": raw_dets}

    def postprocess(self, result: dict) -> list:
        t0 = time.time()
        bboxes = sorted(
            [
                [
                    d["bbox"][0],
                    d["bbox"][1],
                    d["bbox"][2],
                    d["bbox"][3],
                    d["confidence"],
                    d["class_id"],
                ]
                for d in result["raw_dets"]
            ],
            key=lambda x: (x[1], x[0]),
        )
        print(f"[DetectionHandler] postprocess {int((time.time() - t0) * 1000)} ms")
        return [{"predictions": [{"detections": bboxes}]}]

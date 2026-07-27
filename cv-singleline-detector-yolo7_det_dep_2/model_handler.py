"""
model_handler.py
----------------
TorchServe handler for the 2-model YOLOv7 pipeline.

Two request modes (selected via "model_name" field in request body):
  "detection"    — Full shelf image → Stage1Detection → bboxes + LabelRecords
  "segmentation" — Strip images     → Stage2Segmentation → seg masks

Detection request:
  { "instances": [{ "model_name": "detection", "file": "<base64-image>" }] }

Segmentation request:
  { "instances": [
      { "model_name": "segmentation", "strip_id": 0, "file": "<base64-image>" },
      { "model_name": "segmentation", "strip_id": 1, "file": "<base64-image>" }
  ]}
"""

import base64
import io
import os
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# TorchServe unpacks .mar contents into a temp dir at runtime.
# __file__ resolves to that temp dir, so all bundled files/packages are siblings.
# We do NOT add yolov7/ or yolov7-seg/ here — each stage injects only its own
# repo root inside load_model() to prevent the two repos shadowing each other.

_UNPACK_DIR = os.path.dirname(os.path.abspath(__file__))
APP_HOME = os.environ.get("APP_HOME", "/app")

for _p in [_UNPACK_DIR, APP_HOME]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Pipeline imports ───────────────────────────────────────────────────────────
# All imports come AFTER the path bootstrap above.

from common_config_gpu import build_gpu_config, ensure_gpu_only_import_paths  # noqa: E402

ensure_gpu_only_import_paths()

from service_pipeline_gpu.label_record import StripInfo  # noqa: E402
from service_pipeline_gpu.stage1_detection import Stage1Detection  # noqa: E402
from codec_gpu import (  # noqa: E402
    decode_image_png,
    base64_png_to_numpy_image,
    label_record_to_dict,
    seg_results_to_dict,
)


class YoloV7Handler(BaseHandler):
    """TorchServe handler — routes requests to Stage1 (detection) or Stage2 (segmentation)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initialized = False
        # Build config once — stages read weights paths, thresholds, device from here
        self.config = build_gpu_config()
        # Stage objects are created here; models are loaded inside initialize()
        self.stage1 = Stage1Detection(self.config)

    # ── TorchServe lifecycle ───────────────────────────────────────────────────

    def initialize(self, context):
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"[Handler] Initializing on {device}  numpy={np.__version__}  torch={torch.__version__}")
        try:
            t0 = time.time()

            print("[Handler] Loading Stage 1 — YOLOv7 detection model …")
            self.stage1.load_model()
            print("[Handler] Stage 1 ready.")

            elapsed = int((time.time() - t0) * 1000)
            print(f"[Handler] All models initialized in {elapsed} ms")
            self.initialized = True

        except Exception as e:
            print(f"[Handler] Initialization failed: {e}")
            traceback.print_exc()
            self.initialized = False

    def handle(self, data, context):
        if not self.initialized:
            return [{"error": "model not initialized"}]
        try:
            model_input = self.preprocess(data)
            if model_input is None:
                return [{"error": "preprocessing failed"}]
            result = self.inference(model_input)
            return self.postprocess(result)
        except Exception as e:
            print(f"[Handler] handle() error: {e}")
            traceback.print_exc()
            return [{"error": str(e)}]

    # ── Preprocess ─────────────────────────────────────────────────────────────

    def preprocess(self, request):
        try:
            instances = request[0]["body"]["instances"]
            model_name = instances[0]["model_name"]

            if model_name == "detection":
                return self._preprocess_detection(instances[0])
            else:
                print(f"[Handler] Unknown model_name: {model_name!r}")
                return None

        except Exception as e:
            print(f"[Handler] preprocess error: {e}")
            traceback.print_exc()
            return None

    def _preprocess_detection(self, instance: dict) -> dict:
        """Decode base64 shelf image → BGR ndarray for Stage1."""
        t0 = time.time()
        image_rgb = base64_png_to_numpy_image(instance["file"])
        # image_bgr = image_rgb[:, :, ::-1].copy()
        print(f"[Handler] Detection preprocess {int((time.time() - t0) * 1000)} ms shape={image_rgb.shape}")
        return {"mode": "detection", "image_rgb": image_rgb}

    # ── Inference ──────────────────────────────────────────────────────────────

    def inference(self, model_input: dict) -> dict:
        mode = model_input["mode"]

        if mode == "detection":
            t0 = time.time()
            raw_dets = self.stage1.run_inference(model_input["image_rgb"])

            print(f"[Handler] Detection preprocess {int((time.time() - t0) * 1000)}")
            return {"mode": "detection", "raw_dets": raw_dets, "records": raw_dets}

        else:
            print(f"[Handler] inference() received unknown mode: {mode!r}")
            return {"mode": "unknown"}

    def postprocess(self, result: dict) -> list:
        mode = result.get("mode")

        if mode == "detection":
            t0 = time.time()
            # [x1, y1, x2, y2, confidence, class_id] sorted top-to-bottom, left-to-right
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
            # label_records = [label_record_to_dict(r) for r in result["records"]]
            print(f"[Handler] Detection postprocess {int((time.time() - t0) * 1000)} ms")
            return [{"predictions": [{"detections": bboxes}]}]

        else:
            return [{"error": f"unknown mode in postprocess: {mode!r}"}]

"""
model_handler_seg.py
--------------------
TorchServe handler — YOLOv7-seg segmentation only.

Request:
  { "instances": [
      { "model_name": "segmentation", "strip_id": 0, "file": "<base64-png>" },
      { "model_name": "segmentation", "strip_id": 1, "file": "<base64-png>" }
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

_UNPACK_DIR = os.path.dirname(os.path.abspath(__file__))
APP_HOME    = os.environ.get("APP_HOME", "/app")

for _p in [_UNPACK_DIR, APP_HOME]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common_config_gpu import build_gpu_config, ensure_gpu_only_import_paths  # noqa: E402
ensure_gpu_only_import_paths()

from service_pipeline_gpu.label_record import StripInfo                   # noqa: E402
from service_pipeline_gpu.stage2_segmentation import Stage2Segmentation   # noqa: E402
from codec_gpu import base64_png_to_numpy_image  # noqa: E402


class YoloV7SegHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initialized = False
        self.config  = build_gpu_config()
        self.stage2  = Stage2Segmentation(self.config)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self, context):
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"[Handler] Initializing on {device}  numpy={np.__version__}  torch={torch.__version__}")
        try:
            t0 = time.time()
            print("[Handler] Loading Stage 2 — YOLOv7-seg segmentation model …")
            self.stage2.load_model()
            print(f"[Handler] Stage 2 ready in {int((time.time()-t0)*1000)} ms")
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

    # ── Preprocess ────────────────────────────────────────────────────────────

    def preprocess(self, request):
        try:
            instances = request[0]["body"]["instances"]
            t0 = time.time()
            strips = []
            for i, inst in enumerate(instances):
                strip_id  = int(inst.get("strip_id", i))
                strip_img = base64_png_to_numpy_image(inst["file"])
                strips.append(StripInfo(
                    strip_index   = strip_id,
                    strip_image   = strip_img,
                    label_records = [],
                ))
            print(f"[Handler] Segmentation preprocess {len(strips)} strips  {int((time.time()-t0)*1000)} ms")
            return {"strips": strips}
        except Exception as e:
            print(f"[Handler] preprocess error: {e}")
            traceback.print_exc()
            return None

    # ── Inference ─────────────────────────────────────────────────────────────

    def inference(self, model_input: dict) -> dict:
        t0          = time.time()
        seg_results = self.stage2.run_inference(model_input["strips"])
        total_dets  = sum(len(r["detections"]) for r in seg_results)
        print(f"[Handler] Segmentation inference {int((time.time()-t0)*1000)} ms  strips={len(model_input['strips'])}  detections={total_dets}")
        return {"seg_results": seg_results}

    # ── Postprocess ───────────────────────────────────────────────────────────

    def postprocess(self, result: dict) -> list:
        t0      = time.time()
        print(f"[Handler] Segmentation postprocess {int((time.time()-t0)*1000)} ms")
        return [{"predictions": [{"segmentation": result}]}]

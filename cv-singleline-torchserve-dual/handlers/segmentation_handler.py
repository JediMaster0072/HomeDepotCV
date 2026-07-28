"""
TorchServe handler — segmentation only (segmenter.mar worker process).

Request:
  { "instances": [
      { "model_name": "segmentation", "strip_id": 0, "file": "<base64-png>" },
      ...
  ]}

Endpoint: POST /predictions/segmenter
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import torch
from ts.torch_handler.base_handler import BaseHandler


class SegmentationHandler(BaseHandler):
    """Loads only Stage2 / yolov7-seg from this MAR's model_dir."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initialized = False
        self.config = None
        self.stage2 = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = Path(properties["model_dir"]).resolve()
        serialized = context.manifest["model"]["serializedFile"]
        weights_path = model_dir / serialized

        sys.path.insert(0, str(model_dir))
        from mar_bootstrap import prepare_model_dir  # noqa: E402

        model_dir = prepare_model_dir(model_dir)
        weights_path = model_dir / serialized

        print(
            f"[SegmentationHandler] model_dir={model_dir} weights={weights_path} "
            f"exists={weights_path.is_file()} cuda={torch.cuda.is_available()} "
            f"common_config={(model_dir / 'common_config_gpu.py').is_file()} "
            f"pipeline_pkg={(model_dir / 'service_pipeline_gpu').is_dir()} "
            f"entries={sorted(p.name for p in model_dir.iterdir())[:30]}"
        )

        try:
            from common_config_gpu import build_gpu_config  # noqa: E402
            from codec_gpu import base64_png_to_numpy_image  # noqa: E402
            from service_pipeline_gpu.label_record import StripInfo  # noqa: E402
            from service_pipeline_gpu.stage2_segmentation import Stage2Segmentation  # noqa: E402

            self._base64_png_to_numpy_image = base64_png_to_numpy_image
            self._StripInfo = StripInfo
            self.config = build_gpu_config(model_dir=str(model_dir))
            if weights_path.is_file():
                self.config["segmentation_weights"] = str(weights_path)

            self.stage2 = Stage2Segmentation(self.config)
            t0 = time.time()
            print("[SegmentationHandler] Loading Stage 2 — YOLOv7-seg …")
            self.stage2.load_model()
            print(f"[SegmentationHandler] Stage 2 ready in {int((time.time() - t0) * 1000)} ms")
            self.initialized = True
        except Exception as e:
            print(f"[SegmentationHandler] Initialization failed: {e}")
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
            print(f"[SegmentationHandler] handle() error: {e}")
            traceback.print_exc()
            return [{"error": str(e)}]

    def preprocess(self, request):
        try:
            body = request[0].get("body") or request[0]
            if isinstance(body, (bytes, bytearray)):
                import json

                body = json.loads(body)
            instances = body["instances"]
            t0 = time.time()
            strips = []
            for i, inst in enumerate(instances):
                strip_id = int(inst.get("strip_id", i))
                strip_img = self._base64_png_to_numpy_image(inst["file"])
                strips.append(
                    self._StripInfo(
                        strip_index=strip_id,
                        strip_image=strip_img,
                        label_records=[],
                    )
                )
            print(
                f"[SegmentationHandler] preprocess {len(strips)} strips  "
                f"{int((time.time() - t0) * 1000)} ms"
            )
            return {"strips": strips}
        except Exception as e:
            print(f"[SegmentationHandler] preprocess error: {e}")
            traceback.print_exc()
            return None

    def inference(self, model_input: dict) -> dict:
        t0 = time.time()
        seg_results = self.stage2.run_inference(model_input["strips"])
        total_dets = sum(len(r["detections"]) for r in seg_results)
        print(
            f"[SegmentationHandler] inference {int((time.time() - t0) * 1000)} ms  "
            f"strips={len(model_input['strips'])} detections={total_dets}"
        )
        return {"seg_results": seg_results}

    def postprocess(self, result: dict) -> list:
        print("[SegmentationHandler] postprocess")
        return [{"predictions": [{"segmentation": {"seg_results": result["seg_results"]}}]}]

# Local Pipeline Modules

This folder contains extracted helpers for `20260604_veb_new_inference_pipeline_local_2_fullimage.py`.

The local script is an R&D/VM-oriented version of the single-line pipeline. It still owns model setup, local file iteration, and high-level orchestration. The reusable logic now lives in smaller modules:

- `detection_tracks.py`: local YOLO detection fallback handling, buffered crop extraction, `ClipTrack` initialization, and clip-item packaging.
- `detection_stage.py`: crop extraction, deblur, denoise, and simple grayscale enhancement.
- `strips.py`: crop grouping, center padding, and strip coordinate bookkeeping.
- `segmentation_stage.py`: local segmentation calls, per-crop mask checks, and masked clip creation.
- `masked_original_builder.py`: full-resolution masked-original image reconstruction for OCR.
- `ocr_stage.py`: local Google OCR wrapper calls, parsed SKU results, raw OCR fallback, and OCR annotation JSON saving.
- `assignment.py`: OCR bbox-to-detection assignment and strip-to-original coordinate mapping.
- `debug_outputs.py`: debug image and final OCR overlay image writing.

The goal is the same as the production-service segregation: keep behavior the same while making each pipeline stage easier to inspect, test, and modify.

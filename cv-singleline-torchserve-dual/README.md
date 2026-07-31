# Single-line detection and segmentation deployment

This folder is the complete TorchServe deployment for the two single-line
computer-vision models. Detection and segmentation live together here for
deployment, but each model keeps its own source, weights, handler, and worker.

For copy-and-paste commands, see [RUN.md](RUN.md).

## Why the repository is organized this way

The original repository had two top-level model folders, and each folder
contained copies of both YOLO projects. That made it difficult to tell which
files belonged to detection or segmentation and increased the risk of Python
import collisions.

The consolidated layout has one clear boundary:

```text
cv-singleline-torchserve-dual/
├── detection/       # best.pt, yolov7/, detection-only pipeline
├── segmentation/    # segmentation.pt, yolov7-seg/, segmentation-only pipeline
├── handlers/        # handlers used by the dual MAR packages
├── packaging/       # per-worker MAR configuration
├── scripts/         # build and run helpers
└── test-fixtures/   # committed smoke-test images and comparison masks
```

`detection/` contains no segmentation model source. `segmentation/` contains
no detection model source.

## Runtime flow

One Docker container runs one TorchServe service with two isolated workers:

- `POST /predictions/detector` accepts a full shelf image.
- `POST /predictions/segmenter` accepts one or more shelf-strip crops.

Both endpoints use inference port `9000`. TorchServe unpacks `detector.mar` and
`segmenter.mar` into separate model directories, preventing their generic
`models` and `utils` package names from colliding.

The calling application remains responsible for creating strip crops and
combining the two responses.

## Model packages

### Detection

[`detection/`](detection/) contains:

- `best.pt` (gitignored model weight)
- `yolov7/`
- `service_pipeline_gpu/stage1_detection.py`
- a detection-only standalone Dockerfile and handler

### Segmentation

[`segmentation/`](segmentation/) contains:

- `segmentation.pt` (gitignored model weight)
- `yolov7-seg/`
- `service_pipeline_gpu/stage2_segmentation.py`
- a segmentation-only standalone Dockerfile and handler

## Build inputs

The dual Docker build packages:

1. `detection/best.pt`, `detection/yolov7/`, and the detection pipeline into
   `detector.mar`
2. `segmentation/segmentation.pt`, `segmentation/yolov7-seg/`, and the
   segmentation pipeline into `segmenter.mar`

The Docker build context is this directory, not the repository root.

## Requirements

The shared deployment uses:

- Python 3.10
- PyTorch 2.7.0 with CUDA 12.8
- torchvision 0.22.0
- NumPy 1.26.4
- OpenCV 4.10.0.82
- SciPy 1.11 through 1.15

These versions were tested with both models on the RTX 2080 Ti host.

## Color space (OpenCV BGR)

Stage1 and Stage2 expect OpenCV **BGR** and convert to RGB internally
(`img[:, :, ::-1]`). Request decode in `detection/codec_gpu.py` and
`segmentation/codec_gpu.py` therefore converts PIL RGB → BGR so TorchServe
matches `cv2.imread` / the local pipeline.

## Tests and exactness

Test inputs and previously generated masks are under
[`test-fixtures/`](test-fixtures/). The segmentation comparison set is kept in
Git so reorganizations can be checked for pixel-level output changes.

### Historical golden masks vs TorchServe (~0.63 IoU)

Comparing Dockerfile TorchServe to saved local `*_pred_mask.jpg` files gave
mean mask IoU ~0.63. That gap is mostly **different prediction sources** (old
pipeline outputs from another host/stack), not bad golden JPEGs and not mainly
JPEG or OCR enlarge. Detection precision also looks low because TorchServe
returns raw boxes while local outputs are pipeline-filtered. Full breakdown:
[LOCAL_VS_TORCHSERVE.md](LOCAL_VS_TORCHSERVE.md).

### Raw-vs-raw serving exactness (right test)

Script: `scripts/raw_vs_torchserve_segmentation.py` — same `segmentation.pt`
and config (`imgsz=608`, conf `0.25`, iou `0.70`) for in-process Stage2 vs
`/predictions/segmenter`.

| When | Local input | Exact masks (40 strips) | Mean IoU |
|------|-------------|-------------------------|----------|
| Before BGR fix | `cv2.imread` BGR | not exact | ~0.90 |
| Before BGR fix | handler PIL RGB (what TorchServe fed) | 40/40 | 1.0 |
| **After BGR fix** | `cv2.imread` BGR | **40/40** | **1.0** |
| **After BGR fix** | handler (codec BGR) | **40/40** | **1.0** |
| After BGR fix | `legacy_rgb` (old path) | 1/40 | ~0.87 |

Serving is pixel-exact against OpenCV after the fix. Old RGB PNG baselines need
refresh when promoting the BGR-fixed image over the live `:9000` container.

First-time SSH setup, weight transfer, and host troubleshooting are documented
in `../TORCHSERVE_SSH_DEPLOY_172.16.20.100.md`.

GPU deployment checkout:
`/data/vaibhav.singh/SingleLine_deployment/cv-singleline-torchserve-dual`.

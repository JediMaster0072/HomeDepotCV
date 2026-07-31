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

## Tests

Test inputs and previously generated masks are under
[`test-fixtures/`](test-fixtures/). The segmentation comparison set is kept in
Git so reorganizations can be checked for pixel-level output changes.

First-time SSH setup, weight transfer, and host troubleshooting are documented
in `../TORCHSERVE_SSH_DEPLOY_172.16.20.100.md`.

# Single-line detection and segmentation deployment

This folder is the complete packaged service for the two single-line
computer-vision models:

1. **Detection** — finds products / regions on a full shelf photo
2. **Segmentation** — draws the exact shape (mask) of labels on strip crops

Both models are deployed together, but each keeps its own code, trained
weights, request handler, and worker process so they do not interfere with
each other.

For copy-and-paste run commands, see [RUN.md](RUN.md).

## Why the repository is organized this way

Originally there were two separate top-level model folders, and each folder
also contained a copy of the other model’s project files. That made it hard
to tell which files belonged to detection vs segmentation, and it created a
risk that Python would load the wrong shared libraries (name collisions on
packages called `models` and `utils`).

This folder puts everything in one clear place:

```text
cv-singleline-torchserve-dual/
├── detection/       # best.pt, yolov7/, detection-only pipeline
├── segmentation/    # segmentation.pt, yolov7-seg/, segmentation-only pipeline
├── handlers/        # handlers used by the dual MAR packages
├── packaging/       # per-worker MAR configuration
├── scripts/         # build and run helpers
└── test-fixtures/   # committed smoke-test images and comparison masks
```

`detection/` has only detection code. `segmentation/` has only segmentation
code. Neither folder contains the other model’s source tree.

## How the running service works

One Docker container runs one TorchServe service (the model server) with two
separate workers:

- `POST /predictions/detector` — send a full shelf image
- `POST /predictions/segmenter` — send one or more shelf-strip crop images

Default docs/scripts use inference port `9000`. On the RTX 2080 Ti host the
current dual container (`hd-dual-gpu`) is on **`12000`** (management `12001`,
metrics `12002`) because `9000` was already in use by another process.
TorchServe unpacks two model packages (`detector.mar` and `segmenter.mar`)
into separate folders so their shared library names (`models` / `utils`) do
not collide.

The calling application still creates the strip crops and combines the two
responses. This service only runs the two models.

## Model packages

### Detection

[`detection/`](detection/) contains:

- `best.pt` — trained detection weights (not stored in Git)
- `yolov7/` — detection model code
- `service_pipeline_gpu/stage1_detection.py` — detection inference steps
- a detection-only standalone Dockerfile and handler (for running detection alone)

### Segmentation

[`segmentation/`](segmentation/) contains:

- `segmentation.pt` — trained segmentation weights (not stored in Git)
- `yolov7-seg/` — segmentation model code
- `service_pipeline_gpu/stage2_segmentation.py` — segmentation inference steps
- a segmentation-only standalone Dockerfile and handler (for running segmentation alone)

## What gets built into the Docker image

The dual Docker build packages:

1. `detection/best.pt`, `detection/yolov7/`, and the detection pipeline into
   `detector.mar`
2. `segmentation/segmentation.pt`, `segmentation/yolov7-seg/`, and the
   segmentation pipeline into `segmenter.mar`

Build from this folder (`cv-singleline-torchserve-dual/`), not from the whole
repository root.

## Software versions used

The shared deployment uses:

- Python 3.10
- PyTorch 2.7.0 with CUDA 12.8
- torchvision 0.22.0
- NumPy 1.26.4
- OpenCV 4.10.0.82
- SciPy 1.11 through 1.15

These versions were tested with both models on the RTX 2080 Ti GPU host.

## Local vs TorchServe: before / after summary

This section explains whether the deployed service matches “local” model
runs, what looked wrong at first, what the real bug was, and what improved
after the fix.

More narrative detail is in [LOCAL_VS_TORCHSERVE.md](LOCAL_VS_TORCHSERVE.md).
Sample test images and masks are under [`test-fixtures/`](test-fixtures/).

### What the tests compared (important)

The full local overhead pipeline and the deployed dual TorchServe service are
not the same end-to-end program. Locally, detection is followed by crop
buffering, strip building, and later post-processing. Deployed dual TorchServe
only loads an image, runs the model, and returns the result.

In the local pipeline (for context — not inside TorchServe):

- detection confidence threshold examples use **0.50**
- each detection box is expanded by **15% on each side** before cropping
  (`bbox_buffer_pct=0.15`)
- both the original box (`orig_bbox`) and the expanded box (`buffered_bbox`)
  are stored in `predictions.json`
- the expanded crop is what gets saved under `strips/`

What our comparison scripts actually used:

| Test | Script | Local side | TorchServe side |
|------|--------|------------|-----------------|
| Detection (golden) | `scripts/benchmark_local_vs_torchserve.py` | `orig_bbox` from `predictions.json` (raw detection box — **not** `buffered_bbox`) | Raw `/predictions/detector` boxes |
| Segmentation (golden) | same script | Saved masks in `strips_viz_orig_mask/*_pred_mask.jpg`; strip inputs from `strips/` (buffered crops) | `/predictions/segmenter` masks |
| Segmentation (fair / exactness) | `scripts/raw_vs_torchserve_segmentation.py` | Same strip image run through local Stage2 | Same strip through `/predictions/segmenter` |

Related helpers:

- `scripts/diagnose_segmentation_gaps.py` — investigate disagreements
- `scripts/compare_segmentation_regression.py` — endpoint vs saved PNG/JSON baselines

So: detection golden compare used the **raw / draw detection boxes**
(`orig_bbox`), not the buffered crop boxes. Segmentation strip inputs were the
buffered crops; the golden mask files were historical raw-pipeline masks, not
OCR-enlarged masks. The fair exactness test compares the same model step only.

### Starting problem

Running both models in one service was hard to trust because each model folder
also carried copies of the other model’s project. That caused unclear file
ownership and Python import collisions on shared names (`models` / `utils`).

This folder fixes that packaging problem with:

- separate workers (`detector.mar` / `segmenter.mar`)
- one container
- one shared software stack (PyTorch 2.7.0+cu128, torchvision 0.22.0,
  NumPy 1.26.4, OpenCV 4.10.0.82)

After packaging was cleaned up, the next question was:

> Does the live TorchServe service return the same answers as running the same
> model weights locally?

### Before: historical golden comparison (misleading target)

We first compared TorchServe outputs to saved “local” pipeline results
(`local_deployed_instance_outputs` / files named like `*_pred_mask.jpg`).

| What we measured | Result | What it meant in plain terms |
|------------------|--------|------------------------------|
| Segmentation mean mask IoU | ~0.63 | On average, masks overlapped only about 63% — current service ≠ old saved local masks |
| Detection matched-box IoU | ~0.93 | When boxes could be paired, they lined up closely (~93% overlap) |
| Detection precision | looked low | The service returns every raw detection box; the old local files already had extra filtering applied |

**IoU** (Intersection over Union) is a 0–1 overlap score: `1.0` means identical
shapes; `0` means no overlap.

These were **not** the main causes of the ~0.63 gap:

- saving masks as JPEG
- OCR-related mask enlargement
- bad golden strip JPEG inputs

**Main cause:** the saved local masks came from an older pipeline run on another
machine / software stack (`raw_prediction` polygons). They were not independent
human ground truth, and they were not produced with today’s Docker runtime.

So that comparison answered “does today match an old run?” — not “is serving
faithful to the model?”

### Before: raw-vs-raw serving test (the right test)

We then ran the fair test: same weights (`segmentation.pt`), same settings
(`imgsz=608`, confidence `0.25`, IoU threshold `0.70`, `max_det=300`), same
strip images — once inside the process (Stage2) and once through the live
endpoint `/predictions/segmenter`.

Script: `scripts/raw_vs_torchserve_segmentation.py`

| How the local side prepared the image | Exact masks | Mean IoU |
|---------------------------------------|-------------|----------|
| Same array the handler fed the model (PIL RGB — red/green/blue channel order) | 40/40 | 1.0 |
| Normal OpenCV load (`cv2.imread`, BGR — blue/green/red channel order) | not exact | ~0.90 |

**What that told us**

- The server was consistent with itself (not random).
- But it was reading colors in the wrong channel order for how the model code
  expects images.
- OpenCV / the local pipeline uses **BGR**. The server was decoding with PIL as
  **RGB**, then the model still flipped channels as if the image were BGR
  (`img[:, :, ::-1]`). Local OpenCV and TorchServe therefore did not match.

### What we changed / added

**Code (first color fix — codecs to BGR)**

- `detection/codec_gpu.py` / `segmentation/codec_gpu.py` — temporarily convert
  PIL RGB → BGR so Stage’s internal `::-1` flip matched OpenCV
- verified 40/40 exact vs OpenCV on that path

**Code (current contract — stages take RGB, no flip)**

- `detection/service_pipeline_gpu/stage1_detection.py` — removed
  `img[:, :, ::-1]` channel flip; Stage1 expects **RGB**
- `segmentation/service_pipeline_gpu/stage2_segmentation.py` — same
- codecs again pass PIL **RGB** through (no BGR convert)
- `handlers/detection_handler.py` — `image_rgb`

**Docs / scripts**

- this README and `LOCAL_VS_TORCHSERVE.md`
- `scripts/raw_vs_torchserve_segmentation.py` — modes `bgr` (OpenCV→RGB) /
  `handler` / `legacy_bgr`
- `scripts/diagnose_segmentation_gaps.py`, `scripts/benchmark_local_vs_torchserve.py`

**Git / GPU host**

- codec-BGR era: commit `29274a9e`
- RGB Stage / no-flip: commit `1efd030d`
- synced under
  `/data/vaibhav.singh/SingleLine_deployment/cv-singleline-torchserve-dual`
- running container: `hd-dual-gpu` on ports **`12000/12001/12002`**

### After: retest stats

**Codec-BGR era** (stages still had `::-1`, codecs returned BGR), 40 strips:

| Mode | Exact masks | Mean IoU | Meaning |
|------|-------------|----------|---------|
| `bgr` (`cv2.imread`) | 40/40 | 1.0 | Matched OpenCV when Stage expected BGR |
| `handler` (codec BGR) | 40/40 | 1.0 | Serving path stable |
| `legacy_rgb` (old path) | 1/40 | ~0.87 | Pre-fix mismatch |

**Current RGB Stage / no-flip** (rebuilt `hd-dual-gpu` on `:12000`), 40 strips:

| Mode | Exact masks | Mean IoU | Meaning |
|------|-------------|----------|---------|
| `bgr` (OpenCV→RGB) | 40/40 | 1.0 | Fair OpenCV path matches serving |
| `handler` (codec RGB) | 40/40 | 1.0 | Serving path stable |

### Side-by-side

| Comparison | Before any color fix | After codec→BGR fix |
|------------|----------------------|---------------------|
| TorchServe vs OpenCV (fair channel order) | ~0.90 IoU, not exact | 1.0 IoU, pixel-exact |
| TorchServe vs its own preprocess | exact, wrong order for OpenCV | exact |
| TorchServe vs historical golden masks | ~0.63 IoU | still not a fair exactness target |

### Current state

**Channel contract now:** Stage1 and Stage2 expect **RGB** and do **not** flip
channels. TorchServe codecs return PIL RGB. Callers that load with OpenCV
(`cv2.imread`, BGR) must convert BGR→RGB before calling Stage locally.

**Deployed now:** local git, GitHub `main`, and the GPU trees match. Live dual
TorchServe is `hd-dual-gpu` on **`http://127.0.0.1:12000`** (detector +
segmenter READY; verified 40/40 exact). Leave other users’ `:9000` processes
alone.

The old golden mask JPGs remain a historical baseline, not proof of serving
drift.

First-time SSH setup, weight transfer, and host troubleshooting are documented
in `../TORCHSERVE_SSH_DEPLOY_172.16.20.100.md`.

# Local pipeline vs TorchServe: exactness

This document explains why local golden outputs and TorchServe disagreed, what
that means for “serving exactness,” and the color-space fix that makes
OpenCV-style local inference match TorchServe.

For build/run commands, see [RUN.md](RUN.md). For package layout, see
[README.md](README.md).

## Initial problem

The original repo had two top-level model trees, and each tree carried copies of
both YOLO projects (`yolov7` and `yolov7-seg`). That created:

- unclear ownership of detection vs segmentation files
- Python import collisions on generic package names (`models`, `utils`) when
  both workers lived in one process or shared a `sys.path`
- hard-to-trust dual TorchServe packaging

The dual package under `cv-singleline-torchserve-dual/` fixes that by nesting
task-specific trees (`detection/`, `segmentation/`), shipping two MAR workers
(`detector.mar`, `segmenter.mar`) in one container, and aligning the shared
runtime stack (PyTorch 2.7.0+cu128, torchvision 0.22.0, NumPy 1.26.4,
OpenCV 4.10.0.82).

After that restructure, the open question was: **does TorchServe return the same
pixels/boxes as running the same weights locally?**

## What we compared first

Fair IoU script: `scripts/benchmark_local_vs_torchserve.py`

| Side | Source |
|------|--------|
| Local “ground truth” | `local_deployed_instance_outputs` masks / detections from an older overhead pipeline run |
| TorchServe | Current Dockerfile dual image, same strip / shelf inputs from `Golden_Dataset_overhead_eval_orig` |

Results (order of magnitude):

- Detection: matched-box IoU ~0.93, but precision looked low because TorchServe
  returns raw detector boxes while local outputs are pipeline-filtered tracks
- Segmentation: mean mask IoU ~0.63 vs historical local `*_pred_mask.jpg` files

Those numbers do **not** mean the golden JPEG strips are corrupt. They mean the
comparison target was wrong for measuring serving exactness.

## Where the discrepancies come from

### 1. Different prediction sources (main issue)

On a median strip, reconstructing the local mask from `predictions.json`
polygons matched the saved local JPG mask at IoU = 1.0. That same local raw
mask vs current TorchServe was ~0.64.

So `strips_viz_orig_mask/*_pred_mask.jpg` is a saved view of an **old local
pipeline’s** `raw_prediction` polygons, not an independent annotation. Those
outputs were also produced on another host/stack
(`/home/saivijaay.vk/CV/...` in `results.txt`), so weights/runtime likely
differ from today’s `torch==2.7.0+cu128` image.

### 2. Not mainly JPEG

Most sampled local masks have many gray levels (JPEG), but JPEG round-tripping
a TorchServe mask did not change IoU. After thresholding, JPEG is a minor
factor for this comparison.

### 3. Not mainly OCR post-processing

The processor enlarges masks for OCR (`scale_w=1.5`, `scale_h=1.35`), but saved
local masks match **raw** polygons, not the enlarged postprocessed ones.
Applying that enlarge to TorchServe masks made IoU worse, not better.

### 4. Empty-mask edge cases

A few strips are empty locally while TorchServe still fires (or vice versa).
Those create IoU = 0 / trivial IoU = 1 cases, but they are not the bulk of the
~0.63 mean.

### 5. Detection mismatch is a different kind

Local detection ground truth is pipeline-filtered tracks (~0.25 conf, kept
classes). TorchServe returns raw detector boxes, so count/precision gaps are
expected even when matched boxes are close (~0.93 IoU).

## Is it the model or the data?

| Factor | Role |
|--------|------|
| Golden strip / shelf images | Fine as shared inputs |
| Local saved masks as “ground truth” | Misleading — prior pipeline outputs, not independent GT |
| Current Dockerfile model/runtime | Experimental side; differs from the historical local run |
| True annotation GT (SKU polygons) | Exists in golden JSON/CSV; not used for the mask IoU above |

**Bottom line for the ~0.63 IoU:** old local pipeline predictions and current
Dockerfile TorchServe predictions disagree. That is not evidence that the golden
images are bad, and it is not by itself a serving bug.

## Raw-vs-raw serving exactness (the right test)

Script: `scripts/raw_vs_torchserve_segmentation.py`

Run the **same** strip through in-process Stage2 and TorchServe with the same
`segmentation.pt` and config (`imgsz=608`, `conf=0.25`, `iou=0.70`,
`max_det=300`) inside the same Docker image.

Findings before the color-space fix:

| Local Stage2 input | vs TorchServe |
|--------------------|---------------|
| Exact array the handler fed Stage2 (PIL RGB) | **Pixel-exact** (e.g. 40/40, IoU 1.0) |
| `cv2.imread` BGR (OpenCV / local convention) | IoU ~0.90, not exact |

So TorchServe was **deterministic and bit-stable** relative to its own
preprocess. It did **not** match the OpenCV BGR contract that Stage1/Stage2
were written for.

## RGB / BGR bug and fix

### Contract

Both stages assume OpenCV BGR and convert to RGB inside preprocess:

```text
letterbox(image_bgr) → img[:, :, ::-1]  # BGR → RGB for the network
```

### Bug

Handlers decoded request images with PIL (`Image.open` → `np.array`), which
yields **RGB**, then passed that array into Stage2/Stage1 unchanged. The
internal `::-1` then treated RGB as if it were BGR and swapped channels again
the wrong way for OpenCV callers.

### Fix

`detection/codec_gpu.py` and `segmentation/codec_gpu.py` now convert PIL RGB →
BGR after decode so TorchServe matches `cv2.imread` / local pipeline
convention. Detection handler variable names use `image_bgr` accordingly.

Verified on the rebuilt dual test image (`hd-dual-gpu-restructured-test`,
ports `12000–12002`), 40 strips, same `segmentation.pt` / config:

| `--color-mode` | Exact masks | Mean IoU | Meaning |
|----------------|-------------|----------|---------|
| `bgr` (`cv2.imread`) | **40/40** | **1.0** | TorchServe matches OpenCV convention |
| `handler` (current codec BGR) | **40/40** | **1.0** | Serving preprocess is deterministic |
| `legacy_rgb` (old PIL RGB path) | 1/40 | ~0.87 | Pre-fix mismatch reproduced |

Note: older PNG regression baselines captured under the buggy RGB preprocess
will **not** match this fixed image bit-for-bit. Refresh baselines after
promoting the BGR-fixed build.

## How to improve comparisons going forward

1. **Raw-vs-raw exactness** — same weights + config, in-process Stage vs
   TorchServe (`scripts/raw_vs_torchserve_segmentation.py`). This answers
   “is serving exact?”
2. **Quality vs truth** — use golden annotation polygons /
   `evaluate_overhead_pipeline.py`. Treat historical `*_pred_mask.jpg` only as
   a baseline, not GT.
3. **TorchServe PNG baselines** — once Docker is trusted, keep PNG/npz
   masks and use pixel-exact regression (as with the committed temp-strip
   fixtures).
4. **Fair detection** — filter TorchServe to local conf/classes, or export local
   raw detector boxes before pipeline filtering.
5. **No JPEG for exactness** — store PNG/npz masks when pixel equality matters.

## Re-verify after the color fix

```bash
# Inside the dual container (or a GPU host with the package + live segmenter):
python3 scripts/raw_vs_torchserve_segmentation.py \
  --seg-package /app/segmentation \
  --strips-root /path/to/strips \
  --segmenter-url http://127.0.0.1:9000/predictions/segmenter \
  --output-dir /tmp/raw_vs_bgr \
  --color-mode bgr \
  --limit 40
```

Expect `serving_is_exact: true` for `bgr` and `handler` after rebuilding MARs
with the updated codecs.

# Overhead pipeline evaluation

Run from `cv-singleline-processor-CV-1757`:

```bash
python scripts/golden_dataset/evaluate_overhead_pipeline.py
```

The command reads the current golden truth CSV, source image/JSON dataset, and
per-image `predictions_json_export` directory. It writes:

- `summary_metrics.csv`: overall detection, crop, segmentation, and interim OCR metrics
- `class_metrics.csv`: one-to-one parent-object detection precision/recall by class
- `detection_details.csv`: parent-object TP/FP/FN plus raw and buffered geometry
- `pipeline_details.csv`: one row per ground-truth region with stage outcomes
- `image_metrics.csv`: per-image health counts
- `failure_overlays/`: highest-failure images with GT, raw OD, and buffered boxes
- `evaluation_report.md`: human-readable report

Default output:

```text
research_outputs/overhead_pipeline_evaluation/
```

Detection uses confidence-ordered, class-aware, one-to-one matching at IoU
0.5 against parent-object polygons (`RDC`, `Pallet`, etc.). Buffered geometry
is measured against the same OD match and cannot change TP/FP/FN.

SKU-stage evaluation is separate. A prediction is eligible when the
same-class SKU-label center lies inside its buffered crop; eligible pairs are
assigned one-to-one. The report includes both any-track center coverage and
strict one-to-one track matching so collisions remain visible.

Segmentation values are region-overlap proxies because the current golden
polygons identify SKU label regions rather than audited pixel-level
segmentation masks. OCR accuracy only includes reviewed, scorable 6- or
10-digit expected SKUs and is explicitly marked interim until review is
complete.

Useful overrides:

```bash
python scripts/golden_dataset/evaluate_overhead_pipeline.py \
  --iou-threshold 0.5 \
  --max-failure-overlays 40 \
  --output-dir ../research_outputs/overhead_pipeline_evaluation
```

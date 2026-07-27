# Streamlit Model Review Update — July 21, 2026

## What changed

The existing Streamlit link now contains two tabs:

1. **Annotation Team** — the focused SKU ground-truth workflow.
2. **Model Review** — a technical comparison of ground truth and pipeline output.

The annotation tab does not show segmentation or model-debug details. Reviewers can
continue entering expected SKUs, marking `X`, and saving scorable/non-scorable labels
without interacting with the technical review screen.

## Model Review tab

For each image and selected detection, the tab shows six aligned views:

| Ground truth | Raw model output | Processed model output |
| --- | --- | --- |
| Ground Truth OD | Prediction Raw OD (`orig_bbox`) | Prediction Buffered OD (`buffered_bbox`) |
| Ground Truth SKU Region | Raw Segmentation (`raw_prediction`) | Postprocessed Segmentation (`postprocessed_minAreaRect`) |

Below the images it shows:

- Ground-truth SKU
- Predicted SKU from `ocr_words[].text`
- Correct/incorrect/unreviewed status
- Object-detection IoU
- Detection confidence
- Segmentation-found status

## Prediction input

The app now reads one JSON per image from:

`/Users/avinash.patel/Downloads/HomeDepotCV/predictions_json_export/`

The JSON filename is paired directly with the source image:

`<image stem>.json` → `<image stem>.jpg`

This removes the need to infer an image from bounding boxes. The app uses the fields
documented in `PREDICTIONS_JSON_SCHEMA.md`:

- `orig_bbox`
- `buffered_bbox`
- `segmentation.raw_prediction.original_image`
- `segmentation.postprocessed_minAreaRect.original_image`
- `ocr_words[].text`
- `ocr_words[].original_bbox`

## Shared ground truth

Both tabs read the same `golden_sku_truth.csv`. Labels saved by the annotation team
therefore become available to the Model Review tab immediately. The existing sync
script can still copy those values into the source annotation JSON files.

## Deployment

The deployment script now synchronizes the complete `predictions_json_export`
directory to the 5090 host while code-only mode continues to preserve the remote
annotation CSV.

## Validation

- 107 per-image prediction JSON files were discovered.
- 672 OCR predictions were matched to ground-truth regions.
- 12 automated tests passed.
- A full Streamlit smoke test verified both tabs and all six comparison panels.


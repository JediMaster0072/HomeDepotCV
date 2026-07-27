# Annotation tool handoff — 2026-07-23

## Start the shared tool

The deployed team instance runs at:

```text
http://172.16.20.108:8503
```

Corporate VPN access is required.

## Main code

```text
cv-singleline-processor-CV-1757/scripts/golden_dataset/streamlit_expected_sku_review.py
```

Supporting modules:

```text
cv-singleline-processor-CV-1757/scripts/golden_dataset/model_review.py
cv-singleline-processor-CV-1757/scripts/common/crop_preprocess.py
cv-singleline-processor-CV-1757/scripts/common/golden_shapes.py
cv-singleline-processor-CV-1757/scripts/common/paths.py
cv-singleline-processor-CV-1757/scripts/common/prediction_suggestions.py
cv-singleline-processor-CV-1757/scripts/common/sku_review.py
```

## Non-image input/state files

```text
research_outputs/annotation_navigation_debug/golden_sku_truth.remote_latest.csv
research_outputs/annotation_navigation_debug/reviewer_image_assignments.remote_latest.json
research_outputs/golden_dataset_local_tests/review_batch_images.txt
```

- `golden_sku_truth.remote_latest.csv` is the latest remote annotation table.
- `reviewer_image_assignments.remote_latest.json` stores reviewer membership and whole-image assignments.
- `review_batch_images.txt` is the optional current-batch image manifest.

Images, Drona JSONs, and prediction JSONs are intentionally excluded from the
handoff ZIP.

## Navigation behavior

- Saving prefers the next eligible crop in the current image.
- If the reviewer entered the image from a later crop, navigation wraps to any
  remaining earlier crop before advancing to another image.
- Previous/next image navigation uses stable region keys.
- Explicit image navigation switches the annotation filter to `All`, allowing
  completed images to be reopened.

## Verification

The navigation flow was tested against a snapshot of the current remote CSV
and assignment file:

1. Opened Veb's first assigned image.
2. Moved to crop 3 of 3.
3. Saved it.
4. Verified the tool stayed on the same filename and returned to crop 1 of 2.

The automated annotation suite passes 16 tests. The deployed service reports
`active` and HTTP `200`.

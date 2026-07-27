# `predictions.json` — Field Reference

This document explains every key in the `predictions.json` file produced by the
HomeDepot inference pipeline (`cleaned_new_inference_pipeline_local_fullimage.py`
+ the `inference/` modules), so anyone can read and use it without digging
through the code.

For evaluation / final checks the three things that matter most are:

1. **`buffered_bbox`** — the crop region we actually extract and process.
2. **`segmentation.postprocessed_minAreaRect`** — the post-processed segmentation mask (the SKU region actually fed to OCR).
3. **`ocr_words[].text` + `ocr_words[].original_bbox`** — the OCR/SKU parsing result and where it sits on the original image.

Everything else is intermediate bookkeeping used to produce those three.

---

## 1. High-level structure

`predictions.json` is a **JSON array**. Each element is one **`ClipTrack`** —
i.e. one object-detection (OD) box that survived filtering, together with all of
the segmentation and OCR information derived from it.

```json
[
  { /* ClipTrack for detection 0 */ },
  { /* ClipTrack for detection 1 */ },
  ...
]
```

One image → many `ClipTrack` entries (one per detected label region).

> Source: each `ClipTrack` dataclass is serialized with `dataclasses.asdict(...)`
> in the pipeline's `__main__`. The dataclass is defined in `common_config.py`.

---

## 2. Coordinate systems (read this first)

Different keys live in **different coordinate spaces**. Getting this wrong is the
most common mistake when consuming the file.

| Space | Origin (0,0) | Used by |
|-------|--------------|---------|
| **Original image** | Top-left of the full input photo | `orig_bbox`, `buffered_bbox`, `ocr_words[].original_bbox`, `segmentation.*.original_image` |
| **Strip-local** | Top-left of the horizontal "strip" that clips were packed into for segmentation | `strip_bbox`, `crop_strip_bbox` |
| **Crop-local** | Top-left of *this* clip's buffered crop | `segmentation.*.crop_local` |

Rule of thumb: if you want to draw something back on the **original photo**, use
the `*_bbox` fields in original-image space and `segmentation.*.original_image`.

---

## 3. The `BoundingBox` sub-object

Several keys are `BoundingBox` objects with this shape:

```json
{
  "x1": 3154, "y1": 2514,   // top-left  (inclusive)
  "x2": 3353, "y2": 2618,   // bottom-right (exclusive)
  "confidence": 0.855,
  "class_id": 1,
  "class_name": "RDC"
}
```

| Key | Meaning |
|-----|---------|
| `x1, y1` | Top-left corner |
| `x2, y2` | Bottom-right corner |
| `confidence` | Detection confidence carried from the OD model (1.0 for OCR word boxes) |
| `class_id` / `class_name` | Class of the box (see class table below) |

**OD classes** (`class_id` → `class_name`):

| id | name |
|----|------|
| 0 | Pallet |
| 1 | RDC |
| 2 | Printed_on_Box |
| 3 | Handwritten |
| 4 | Multiline_Label *(skipped by the pipeline)* |
| 5 | Other |

---

## 4. `ClipTrack` keys (top level of each array element)

⭐ = most important for final checks.

| Key | Type | Space | Meaning |
|-----|------|-------|---------|
| `det_id` | int | — | Index of this detection within the image. |
| `class_id` / `class_name` | int / str | — | OD class of the detected region. |
| `confidence` | float | — | OD detection confidence. |
| `orig_bbox` | BoundingBox | Original image | The **raw** OD box, exactly as the detector output it (no buffer). |
| ⭐ `buffered_bbox` | BoundingBox | Original image | `orig_bbox` expanded by `BBOX_BUFFER_PCT` (15%) on each side, clamped to image bounds. **This is the crop that is actually extracted and sent to segmentation + OCR.** |
| `clip_h` / `clip_w` | int | — | Height/width (px) of the buffered crop. |
| `pad_h` / `pad_w` | int | — | Size of this clip's slot in the strip (group-local max height/width across the ≤5 clips packed together). |
| `pad_offset_x` / `pad_offset_y` | int | Strip slot | Where the crop sits inside its (center-padded) slot. |
| `strip_index` | int | — | Which strip this clip was packed into (clips are grouped ≤5 per strip). |
| `strip_slot` | int | — | Position of this clip within its strip (0-based, left→right). |
| `strip_bbox` | BoundingBox | Strip-local | The **full padded slot** occupied by this clip in the strip (includes padding). |
| `crop_strip_bbox` | BoundingBox | Strip-local | The **actual crop region** (excluding padding) inside the strip. Masks are sliced against this. |
| `seg_found` | bool | — | Whether a valid segmentation region was found for this clip (passed the white-pixel check). |
| `ocr_found` | bool | — | Whether at least one OCR word was assigned to this clip. |
| `metadata` | dict | — | Misc debug info, e.g. `{"strip_mask": "0_pred_mask.jpg"}` = the saved strip-mask debug image for this clip. |
| ⭐ `segmentation` | dict | mixed | Full segmentation geometry (raw + post-processed). See §5. |
| ⭐ `ocr_words` | list | Original image | The OCR/SKU results assigned to this clip. See §6. |

### How the intermediate coordinates relate

```
orig_bbox   ──(+15% buffer, clamp)──►  buffered_bbox        (original image)
buffered_bbox ──(extract crop)──►  crop  ──(center-pad into slot)──►  strip
   crop's location inside the strip  =  crop_strip_bbox               (strip-local)
   crop_strip_bbox top-left  ≡  buffered_bbox top-left  (1:1, no scaling)
```

That last equivalence is why `segmentation` geometry can be mapped straight back
to the original image just by adding `buffered_bbox.x1 / y1`.

---

## 5. The `segmentation` block ⭐

Records the SKU segmentation mask for this clip in a JSON-friendly (polygon /
rotated-rectangle) form. It captures **two versions** of the mask, each in **two
coordinate spaces**.

```json
"segmentation": {
  "found": true,
  "crop_size": [257, 134],        // [width, height] of the buffered crop
  "white_pixels": 14912,          // # of mask pixels (post-processed) inside the crop
  "postprocessed_minAreaRect": {  // the mask ACTUALLY used for OCR
    "crop_local":     { "polygons": [...], "rotated_rects": [...] },
    "original_image": { "polygons": [...], "rotated_rects": [...] }
  },
  "raw_prediction": {             // the raw SEG mask BEFORE post-processing
    "crop_local":     { "polygons": [...], "rotated_rects": [...] },
    "original_image": { "polygons": [...], "rotated_rects": [...] }
  }
}
```

| Key | Meaning |
|-----|---------|
| `found` | Same as `seg_found` — was a usable mask found for this clip. |
| `crop_size` | `[width, height]` of the buffered crop (the coordinate frame for `crop_local`). |
| `white_pixels` | Count of active mask pixels (post-processed) within the crop. Useful as a quick "how much got segmented" signal. |
| ⭐ `postprocessed_minAreaRect` | Geometry of the **enlarged** mask — i.e. after `cv2.minAreaRect` + the 1.5×(width)/1.35×(height) expansion in `process_binary_mask_with_rotation`. **This is what defines the SKU region that OCR sees.** |
| `raw_prediction` | Geometry of the **raw** SEG model output, *before* the minAreaRect enlargement. Handy for debugging how much the post-processing grew the region. |
| `crop_local` | Coordinates relative to the buffered crop's top-left. |
| `original_image` | Same geometry mapped onto the full original photo (= `crop_local` + `buffered_bbox` top-left). |

### `polygons` and `rotated_rects`

Each of `crop_local` / `original_image` contains:

```json
{
  "polygons": [
    [ [x, y], [x, y], ... ]      // one closed contour of the mask region
  ],
  "rotated_rects": [
    {
      "center": [cx, cy],        // rotated-rect center
      "size": [w, h],            // rotated-rect width, height
      "angle": 1.33,             // rotation angle in degrees
      "box_points": [            // the 4 corners of the rotated rect
        [x, y], [x, y], [x, y], [x, y]
      ]
    }
  ]
}
```

- `polygons` — raw contour points (exact mask outline).
- `rotated_rects` — the tight rotated bounding box of each contour (`center`,
  `size`, `angle`, plus explicit `box_points`).

> **Clipping note:** the mask geometry is computed *after* the mask is sliced to
> the clip's crop region. So the post-processed box is **cut to the buffered-crop
> boundary** — the enlargement never extends past the buffered crop onto the
> original image. Compare `postprocessed_minAreaRect.original_image` box points
> against `buffered_bbox` to see this directly.

---

## 6. The `ocr_words` block ⭐

The OCR / SKU-parsing results assigned to this clip. Empty list `[]` means no OCR
word landed inside this detection.

```json
"ocr_words": [
  {
    "text": "1000004778",         // normalized SKU string
    "raw_text": "1000004778",     // original OCR text before normalization
    "det_id": 0,                  // det_id of the ClipTrack this word belongs to
    "class_id": 1,
    "class_name": "RDC",
    "source": "seg+google_ocr",   // how it was produced
    "strip_bbox": null,           // not used in the full-image flow
    "original_bbox": {            // where the word sits on the ORIGINAL image
      "x1": 3168, "y1": 2574, "x2": 3335, "y2": 2614,
      "confidence": 1.0, "class_id": 0, "class_name": ""
    },
    "confidence": 1.0
  }
]
```

| Key | Meaning |
|-----|---------|
| ⭐ `text` | The **normalized SKU** string — use this for accuracy checks. |
| `raw_text` | The original OCR text before SKU normalization. |
| `det_id` / `class_id` / `class_name` | Identify the parent `ClipTrack`. |
| `source` | Origin tag, e.g. `"seg+google_ocr"`. |
| `strip_bbox` | `null` in the current full-image pipeline (legacy field). |
| ⭐ `original_bbox` | The word's bounding box in **original-image** coordinates (for drawing / matching against ground truth). |
| `confidence` | OCR confidence (currently `1.0`). |

---

## 7. Quick recipe for "final checks"

To evaluate an image:

1. For each `ClipTrack`, take the region of interest = **`buffered_bbox`**.
2. The SKU region actually read = **`segmentation.postprocessed_minAreaRect.original_image`**.
3. The predicted SKU(s) = every **`ocr_words[].text`** across all tracks, located
   at **`ocr_words[].original_bbox`**.

Compare those predicted SKUs (and/or boxes) against your ground truth.

Fields like `strip_*`, `pad_*`, `crop_strip_bbox`, `raw_prediction`, and
`metadata` are intermediate/debug and usually not needed for scoring.

---

## 8. Where each key is produced (for maintainers)

| Stage | File | Produces |
|-------|------|----------|
| Detection (Steps 1–2) | `inference/detection_inference.py` | `det_id`, `class_*`, `confidence`, `orig_bbox`, `buffered_bbox`, `clip_h/w` |
| Strip build (Step 4.1) | `inference/strip_processor.py::create_strip` | `pad_*`, `strip_index`, `strip_slot`, `strip_bbox`, `crop_strip_bbox`, `metadata.strip_mask` |
| Segmentation + post-proc | `inference_local.py::process_binary_mask_with_rotation`, recorded in `strip_processor.py::_record_clip_segmentation` / `_extract_mask_geometry` | `seg_found`, `segmentation.*` |
| OCR (Steps 6–7) | `inference/ocr_inference.py` | `ocr_found`, `ocr_words[*]` |
| Serialization | `cleaned_new_inference_pipeline_local_fullimage.py` (`__main__`) | writes `predictions.json` via `asdict(track)` |

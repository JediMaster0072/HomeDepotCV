# Research Findings: Rotation Failures And Empty Shelf Anomalies
This README summarizes the local research artifacts created for two immediate project tasks: researching rotated/upside-down label failures, and thinking through empty-shelf anomaly detection. The work is exploratory and meant to support discussion with Harish/Vaibhav, not to serve as production model output.

## New Files Created
- `research_outputs/RESEARCH_FINDINGS_README.md`: This consolidated summary file.
- `research_outputs/HARISH_VAIBHAV_HYPOTHESIS_SUMMARY.md`: Short hypothesis summary for discussion.
- `research_outputs/ROTATION_AND_EMPTY_ANOMALY_RESEARCH_NOTES.md`: Initial notes on both research tracks.
- `research_outputs/rotation_label_samples/RDC_samples.jpg`: RDC crop contact sheet.
- `research_outputs/rotation_label_samples/Printed_on_Box_samples.jpg`: Printed-on-box crop contact sheet.
- `research_outputs/rotation_label_samples/Pallet_samples.jpg`: Pallet crop contact sheet.
- `research_outputs/rotation_label_samples/Handwritten_samples.jpg`: Handwritten crop contact sheet.
- `research_outputs/rotation_label_samples/sample_manifest.csv`: Metadata for sampled label crops.
- `research_outputs/rotation_label_samples/rotation_review_template.csv`: Full manual-review template.
- `research_outputs/rotation_label_samples/initial_rotation_review_observations.csv`: Small starter review CSV.
- `research_outputs/prototype_rdc_anchor_detection.py`: Prototype script for RDC dark-anchor detection.
- `research_outputs/rdc_anchor_detection/rdc_anchor_detection_overlay.jpg`: Visual output from the RDC prototype.
- `research_outputs/rdc_anchor_detection/rdc_anchor_detection_results.csv`: Per-sample RDC anchor detection results.
- `research_outputs/temporal_empty_analysis/README.md`: README for temporal empty-region artifacts.
- `research_outputs/temporal_empty_analysis/temporal_empty_region_summary.csv`: Parsed `EmptyItem` region summary.
- `research_outputs/temporal_empty_analysis/temporal_empty_region_center_map.jpg`: Coordinate-space map of empty-region centers.
- `temporal_data_overlay/`: Folder containing 15 approximate overlay images, one per temporal capture.
- `temporal_data_overlay/README.md`: README explaining the overlay images and caveats.

## Source Data
### Rotation / Label Data
Source folder: `stratified_output_dataset/`

This dataset contains shelf/product images and YOLO-style label annotations. The rotation research focused on `RDC`, `Printed_on_Box`, `Pallet`, and `Handwritten`. The goal was to inspect cropped examples and decide whether each class has stable visual anchors that could support automatic orientation correction.

### Temporal / Empty Shelf Data
Source folder: `temporal_data/`

Important source file: `temporal_data/Temporal_data_sample.csv`

This CSV contains annotations for multiple captures of the same store/aisle/bay over time. The empty-shelf research focused on rows where `classification == EmptyItem`. The goal was to understand whether empty shelf regions appear in consistent shelf-coordinate locations across the 15 temporal images.

## Rotated / Upside-Down Label Task
### Task Goal
Harish and Vaibhav emphasized that rotated or upside-down labels are a key remaining issue. The research question is: can label orientation be corrected using class-specific anchor points?

An anchor point is a stable visual feature that should appear in a known location when the label is upright. Examples include a dark printed header near the top or a circular mark near the top-left corner.

### Generated Files And Findings
#### `research_outputs/rotation_label_samples/RDC_samples.jpg`
Contact sheet of cropped `RDC` label samples.

Purpose:
- Manually inspect upright vs rotated RDC labels.
- Look for recurring dark anchor regions.
- Identify OCR-risk examples where text is vertical or skewed.

Conclusion:
- `RDC` has rotated examples, but the anchor signal is not clean yet.
- A simple dark-component detector often finds SKU text or dark bands, not necessarily a unique orientation anchor.

#### `research_outputs/rotation_label_samples/Printed_on_Box_samples.jpg`
Contact sheet of cropped `Printed_on_Box` samples.

Purpose:
- Inspect whether the printed label header/band can be used as an orientation anchor.

Conclusion:
- `Printed_on_Box` has one of the clearest anchor signals.
- Many labels contain a dark printed header or band that changes position predictably when rotated.
- This class looks like a strong candidate for quadrant-based orientation correction.

#### `research_outputs/rotation_label_samples/Pallet_samples.jpg`
Contact sheet of cropped `Pallet` samples.

Purpose:
- Inspect whether the dark circular pallet marker can be used as an orientation anchor.

Conclusion:
- `Pallet` has a strong orientation signal.
- Many upright labels show a dark circular mark near the top-left.
- This class is another strong candidate for anchor/quadrant-based correction.

#### `research_outputs/rotation_label_samples/Handwritten_samples.jpg`
Contact sheet of cropped `Handwritten` samples.

Purpose:
- Inspect handwritten/numeric label crops for stable printed anchors.

Conclusion:
- `Handwritten` appears weakest for anchor-based correction.
- Many samples lack a consistent printed anchor.
- A better approach may be OCR across multiple rotations or a small orientation classifier.

#### `research_outputs/rotation_label_samples/rotation_review_template.csv`
Spreadsheet-style template for manual review. It gives one row per sample so a reviewer can fill in orientation, anchor visibility, and notes. This is the broader annotation template, not just the small initial reviewed subset.

#### `research_outputs/rotation_label_samples/initial_rotation_review_observations.csv`
Small manually inspected starter CSV. It captures a few obvious examples per class and records class, sample index, orientation, anchor visibility, OCR risk, and notes. This file is intentionally small and not a full annotation pass.

#### `research_outputs/prototype_rdc_anchor_detection.py`
Prototype script for detecting a likely dark anchor in `RDC` crops.

Purpose:
- Load RDC examples from `stratified_output_dataset`.
- Crop RDC labels using YOLO annotations.
- Threshold dark regions in each crop.
- Find connected dark components.
- Pick the largest/densest component as a possible anchor.
- Assign the selected component to a quadrant.
- Generate a visual overlay and CSV for review.

Conclusion:
- The prototype is useful for surfacing candidate dark regions.
- It is not yet reliable as an RDC orientation corrector because it often selects text strokes or dark SKU regions.
- The next RDC step would be stricter class-specific filtering or a better definition of what the RDC anchor should be.

#### `research_outputs/rdc_anchor_detection/rdc_anchor_detection_overlay.jpg`
Visual output from the RDC anchor prototype. It shows each RDC crop with the selected dark component boxed in red, plus quadrant crosshairs for review.

Conclusion:
- The method detects dark components consistently.
- Many detections are not guaranteed to be true orientation anchors.

#### `research_outputs/rdc_anchor_detection/rdc_anchor_detection_results.csv`
CSV output from the RDC anchor prototype. It stores whether an anchor was found, its quadrant, box size, area, fill ratio, aspect ratio, source split, source image, and label line. It is useful for sorting/filtering candidate detections and should be paired with visual review from the overlay image.

### Rotation Task Summary
Best candidates for anchor-based orientation correction:
- `Printed_on_Box`
- `Pallet`

Needs more care:
- `RDC`

Least suited for simple anchor correction:
- `Handwritten`

Recommended next step: start with class-specific anchor correction for `Printed_on_Box` and `Pallet`, then revisit `RDC` with a stricter detector. For `Handwritten`, consider OCR over multiple rotations or a learned orientation classifier instead of anchor detection.

## Empty Shelf Anomaly Task
### Task Goal
Vaibhav asked whether empty shelf regions can be treated as anomalies, mentioning AnomalyCLIP-style approaches. The research question is: should empty shelf detection start as generic anomaly detection, or should it first use temporal/shelf-location context?

Current answer: empty shelf detection should first be treated as a conditional, temporal anomaly. An empty-looking patch is only suspicious if a product is expected at that shelf location based on prior images, shelf structure, price tags, or beam context.

## What "Same Shelf Location Changed From Product-Present To Empty" Means
Imagine the same bay is photographed several times over multiple days. At one shelf location:
- Day 1: a product is hanging there.
- Day 2: the same spot still has a product.
- Day 3: the same spot is now blank/empty.

That change is stronger evidence of a true empty-shelf event than simply finding a visually blank patch in one image. Many shelf regions can look empty for harmless reasons: gaps between products, pegboard holes, shadows, shelf hardware, blank packaging, or camera angle changes.

So the better research framing is not only, "Does this patch look anomalous?" Instead, ask: "At this same approximate shelf location, did the visual state change from product-present to empty over time?" This is why temporal comparison should come before broad AnomalyCLIP-style scoring.

## Empty Shelf Generated Files
#### `research_outputs/temporal_empty_analysis/temporal_empty_region_summary.csv`
Structured summary of all parsed `EmptyItem` annotations from `temporal_data/Temporal_data_sample.csv`.

Purpose:
- One row per parsed empty region.
- There are 15 temporal images and 42 parsed empty regions.
- Includes each region's bounding box and center point in shelf-coordinate space.

Important columns:
- `image_idx`: Local index for the temporal image, from 1 to 15.
- `captured_ts`: Capture timestamp.
- `url_tail`: Source image identifier.
- `region_idx`: Empty-region index within that image.
- `x1`, `y1`, `x2`, `y2`: Empty-region bounding box in shelf-coordinate space.
- `cx`, `cy`: Center point of the empty region.
- `x_bin_0p5`, `y_bin_0p5`: Coarse bins used to see whether empty regions recur in similar shelf-coordinate locations.

Caveat:
- This file is complete relative to `Temporal_data_sample.csv`.
- It is not truncated.
- The source CSV itself is a sample dataset, not the full production temporal dataset.

#### `research_outputs/temporal_empty_analysis/temporal_empty_region_center_map.jpg`
Coordinate-space scatter plot of empty-region centers.

Purpose:
- Shows where empty-region centers appear across the 15 temporal images.
- Each dot is one empty region.
- The number next to a dot is `image_idx`, not likelihood or severity.
- Repeated dots near each other suggest the same approximate shelf-coordinate area is empty across multiple captures.

Caveat:
- This is not an overlay on a shelf photo.
- It is an abstract coordinate-space map.
- It should be used to reason about temporal clustering, not exact pixel locations.

Conclusion:
- Empty regions do show repeated coarse-position clusters.
- Strong recurring bins included approximately `(2.0, 2.0)`, `(2.0, 1.5)`, `(2.5, 0.5)`, `(2.5, 2.0)`, and `(2.0, 0.5)`.
- This supports temporal comparison as the first baseline.

#### `research_outputs/temporal_empty_analysis/README.md`
Focused README explaining the temporal empty-region summary CSV and center map. It documents the CSV columns, the coordinate-space map, and the finding that empty regions recur in coarse shelf-coordinate bins.

#### `temporal_data_overlay/`
Folder containing one generated overlay image per temporal capture.

Purpose:
- Show the original shelf images with visual markers placed on top.
- Help build intuition for where the parsed `EmptyItem` regions roughly correspond in each image.

Caveat:
- These overlays are approximate.
- The source annotations are in shelf-coordinate space, not direct image pixel coordinates.
- The overlay uses a simple global coordinate-to-image projection.
- Because of camera angle, perspective, and coordinate-system mismatch, dots/boxes may not line up exactly with the true empty spot.
- These files are useful for intuition, but they are not pixel-perfect ground truth.

#### `temporal_data_overlay/README.md`
Focused README explaining how to read the overlay images. It explains color mapping, labels like `2.3` meaning image 2 empty region 3, and why overlays are approximate.

## Final Research Conclusions
### Rotation
The strongest orientation-correction path is class-specific anchor detection.

Most promising:
- `Printed_on_Box`: dark printed header/band.
- `Pallet`: dark circular marker.

Less clear:
- `RDC`: dark regions exist, but the current detector often finds text/label artifacts rather than a clean anchor.

Weakest:
- `Handwritten`: no reliable printed anchor.

### Empty Shelf Anomaly
Empty shelves should not start as generic image anomaly detection. The better first baseline is:
1. Track approximate shelf locations over time.
2. Identify candidate regions that switch from product-present to empty.
3. Use price tags, beams, and shelf geometry to decide whether a product should be expected there.
4. Then use embeddings or AnomalyCLIP-style methods to score candidate regions.

Main takeaway: AnomalyCLIP may be useful later, but it should be applied to context-selected candidate regions, not blindly over the full shelf image.

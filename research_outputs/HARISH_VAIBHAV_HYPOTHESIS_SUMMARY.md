# Hypothesis Summary For Harish / Vaibhav

## Rotated / Upside-Down Label Failures

Based on the local crop contact sheets, the clearest orientation signals appear in `Printed_on_Box` and `Pallet`. `Printed_on_Box` frequently has a dark printed header/band that moves predictably when the crop is rotated, and `Pallet` often has a dark circular mark near the top-left area when upright. These two classes look like the best first candidates for anchor/quadrant-based correction. `RDC` is still worth testing, but the first dark-component prototype often picks dark text strokes or SKU bands rather than a unique class-specific anchor, so it needs stricter filtering or a better definition of the expected RDC anchor. `Handwritten` appears least suitable for anchor-based correction because it usually lacks a reliable printed anchor; for handwritten labels, an OCR-try-all-rotations fallback or a small orientation classifier may be more realistic.

The quadrant-based approach is feasible when the label class has a stable printed marker. The prototype direction is: crop the detected label, threshold dark connected components, select the likely anchor, determine its quadrant, and rotate the crop until the anchor lands in the expected quadrant. The immediate next step should be a manually reviewed orientation sheet for a small sample of each class, then class-specific anchor rules for `Printed_on_Box` and `Pallet`, followed by a stricter `RDC` experiment.

Generated artifacts:

- `research_outputs/rotation_label_samples/initial_rotation_review_observations.csv`
- `research_outputs/rotation_label_samples/rotation_review_template.csv`
- `research_outputs/rdc_anchor_detection/rdc_anchor_detection_overlay.jpg`
- `research_outputs/rdc_anchor_detection/rdc_anchor_detection_results.csv`
- `research_outputs/prototype_rdc_anchor_detection.py`

## Empty Shelf As Anomaly Detection

The strongest framing is that empty shelf detection should be treated as a conditional anomaly, not as generic "empty-looking pixels." A region is only anomalous if a product is expected there based on price tags, shelf/bay geometry, or previous captures of the same bay. The temporal sample has 15 images from the same bay and 42 parsed empty-region boxes. Empty regions do recur in coarse shelf-coordinate bins, especially around `(2.0, 2.0)`, `(2.0, 1.5)`, and `(2.5, 0.5)`, but there are also outliers. This suggests the right first step is temporal comparison with location gating, not open-ended AnomalyCLIP.

Recommended order: first align regions across the 15 temporal captures using shelf coordinates and beam/price-tag context; then compare candidate empty crops against previous product-present crops from the same approximate location; then try embedding similarity or AnomalyCLIP only after this baseline is established. AnomalyCLIP may still be useful, but it should score candidate regions selected by business context rather than search the entire shelf image independently.

Generated artifacts:

- `research_outputs/temporal_empty_analysis/temporal_empty_region_summary.csv`
- `research_outputs/temporal_empty_analysis/temporal_empty_region_center_map.jpg`

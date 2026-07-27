# Rotation And Empty-Anomaly Research Notes

This note focuses on the two near-term research items from the transcripts: rotated/upside-down single-line label failures, and whether empty shelf spaces can be treated as visual anomalies.

## 1. Rotated / Upside-Down Label Failures

Harish and Vaibhav identified orientation as one of the major remaining failure modes for the single-line pipeline. The current pipeline detects single-line labels, crops them, segments the SKU region, builds a cleaner OCR image, and then sends it to Google OCR. Google OCR can often handle mild rotation or 90-degree text, but it struggles when the label is fully upside down or when blur/occlusion is combined with rotation. This means orientation correction is most valuable if it happens before OCR, either on the detected crop, on the segmented SKU region, or on the masked original image.

The labels to focus on are `RDC`, `Printed_on_Box`, `Pallet`, and `Handwritten`. The local dataset has enough examples to inspect immediately: train labels include 2,083 `Pallet`, 3,324 `RDC`, 8,081 `Printed_on_Box`, and 8,075 `Handwritten` instances; validation labels include 436 `Pallet`, 820 `RDC`, 1,689 `Printed_on_Box`, and 1,685 `Handwritten` instances. I generated crop contact sheets for the first-pass visual review here:

- `research_outputs/rotation_label_samples/Pallet_samples.jpg`
- `research_outputs/rotation_label_samples/RDC_samples.jpg`
- `research_outputs/rotation_label_samples/Printed_on_Box_samples.jpg`
- `research_outputs/rotation_label_samples/Handwritten_samples.jpg`
- `research_outputs/rotation_label_samples/sample_manifest.csv`

The most promising direction is anchor-based orientation correction. From Harish's explanation, `RDC` labels usually have a dark rectangular band that should appear in the upper region of the crop. `Printed_on_Box` labels have a dark square anchor that should appear in a specific quadrant, and `Pallet` labels have a dark circular anchor that should also appear in a consistent quadrant. A simple OpenCV prototype can threshold dark connected components, locate the largest likely anchor, determine its crop quadrant, and rotate the crop until the anchor lands in the expected quadrant. This is likely feasible for `RDC`, `Printed_on_Box`, and `Pallet`.

`Handwritten` is probably harder because it may not have a stable printed anchor. For handwritten labels, a more practical approach may be to rely on OCR confidence across rotations, or use a lightweight orientation classifier trained from manually labeled crops. A simple baseline is to run OCR or digit parsing on 0, 90, 180, and 270 degree rotations and keep the orientation that yields a valid 6-digit or 10-digit SKU with the cleanest OCR result. This is slower than anchor detection but can be useful for debugging or for a fallback path.

The first useful deliverable is not a full model. It is a small orientation-review dataset: crop examples by label class, mark them as upright / rotated 90 / rotated 180 / rotated 270 / unclear, and record whether an anchor is visible. That gives Harish and Vaibhav evidence about how frequent the orientation problem is and which label classes are worth solving first. After that, the next deliverable can be an anchor-detection prototype for `RDC`, `Printed_on_Box`, and `Pallet`.

## 2. Empty Shelf As Anomaly Detection

Vaibhav asked whether empty regions can be treated as anomalies, mentioning AnomalyCLIP-style approaches. The key idea is that "empty" is not just a visual class; it is contextual. A blank shelf region is only a problem if a product is expected there, usually because a nearby price tag, shelf position, or historical image indicates that product should be present. So the better framing is: empty shelf detection is an anomaly conditioned on shelf location, price-tag association, and historical product presence.

The local `temporal_data` folder is useful for thinking about this because it contains 15 images for the same store/aisle/bay over time. The CSV has 301 rows: 271 `sku_selling`, 15 `EmptyItem`, and 15 `Beams`. Every temporal image has at least one `EmptyItem` annotation, with detected empty counts ranging from 1 to 7. This means the immediate local task is to understand whether the same regions repeatedly appear empty, whether empty regions move over time, and whether previous days show products in those locations.

There are three feasible research directions. The simplest is image-difference or temporal comparison: align the same bay across days, crop candidate empty regions, and compare them to the same region on previous days. If historical crops contain products and today's crop is visually blank, that supports an out-of-stock anomaly. The second direction is embedding-based: use a vision embedding model to compare today's candidate empty crop against historical product-present crops for the same bay/price-tag region. The third direction is open-vocabulary anomaly detection, such as AnomalyCLIP-style scoring, where prompts like "empty shelf space" and "shelf with product" can be used to score suspicious regions.

The main risk is false positives. Empty-looking shelf space may be normal background, beam area, label area, or an occluded region. Therefore, any anomaly approach should be gated by existing detections: price tags, shelf beams, product regions, and temporal history. The strongest version of the problem is not "find all empty-looking pixels"; it is "for a known shelf/price-tag position where a product is expected, decide whether the current region is anomalously empty compared with historical evidence."

For now, this is more of a research/design task than a coding deliverable. The useful output to bring back to Vaibhav is a short proposal: define empty as a conditional anomaly, use temporal images as historical context, compare empty-region crops to previous product-present crops, and only explore AnomalyCLIP after establishing simple temporal and embedding baselines.

## Recommended Next Steps

First, manually inspect the four rotation contact sheets and make a small spreadsheet or CSV with class, sample index, orientation, anchor visibility, and notes. Second, prototype anchor detection on `RDC`, because the dark rectangular band is likely the easiest anchor to detect. Third, inspect the 15 temporal images side by side with `Temporal_data_sample.csv` and identify whether empty regions correspond to consistent shelf positions over time. Fourth, write a short hypothesis summary for Harish/Vaibhav: which label class has the clearest orientation signal, and whether empty-shelf anomaly detection should start with temporal comparison before AnomalyCLIP.

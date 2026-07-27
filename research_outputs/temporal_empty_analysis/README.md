# Temporal Empty Shelf Analysis

This folder contains the first-pass analysis for the empty-shelf anomaly detection research task. The goal is to understand whether empty shelf regions in the temporal sample appear in consistent shelf positions over time, before trying a more general anomaly model like AnomalyCLIP.

## Files

### `temporal_empty_region_summary.csv`

This CSV is a structured summary of the `EmptyItem` annotations from `temporal_data/Temporal_data_sample.csv`.

Each row represents one parsed empty region from one temporal image. There are 15 temporal images in the sample and 42 parsed empty regions.

Important columns:

- `image_idx`: Local index assigned to the temporal image, from 1 to 15.
- `captured_ts`: Timestamp for when that image was captured.
- `url_tail`: Final identifying part of the source image URL.
- `region_idx`: Empty-region index within that image.
- `x1`, `y1`, `x2`, `y2`: Bounding box corners for the empty region in shelf-coordinate space.
- `cx`, `cy`: Center point of the empty region.
- `w`, `h`: Width and height of the empty region.
- `x_bin_0p5`, `y_bin_0p5`: Coarse 0.5-unit bins used to check whether empty regions recur in approximately the same shelf location.

The binned center columns are useful because exact coordinates vary slightly across annotations, but repeated empty areas should still cluster into similar coarse bins.

### `temporal_empty_region_center_map.jpg`

This image plots the center point of every parsed empty region across the 15 temporal images.

How to read it:

- Each dot is one empty-region center.
- The number next to each dot is the `image_idx` from the CSV.
- Repeated dots near the same area suggest that empty regions are happening in consistent shelf positions.
- Isolated dots are outliers or one-off empty detections.

This map is not a literal photo of the shelf. It is a coordinate-space visualization that helps answer: "Do empty regions recur in the same shelf areas over time?"

## Current Finding

The empty regions do show repeated coarse-position clusters. The strongest bins from the first pass were around:

- `(2.0, 2.0)`
- `(2.0, 1.5)`
- `(2.5, 0.5)`
- `(2.5, 2.0)`
- `(2.0, 0.5)`

This supports the idea that empty shelf detection should start with temporal comparison and shelf-location consistency, not with a fully open-ended anomaly model.

## Recommended Research Direction

Treat empty shelf detection as a conditional anomaly:

An area is suspicious only if a product is expected there based on shelf position, nearby price tags, beam context, or prior images of the same bay.

Suggested progression:

1. Use shelf-coordinate bins to find locations that repeatedly appear empty or switch between product-present and empty.
2. Compare the same shelf location across multiple temporal images.
3. Add context from price tags and beams so the model knows whether a product should exist there.
4. Only then try visual embeddings or AnomalyCLIP-style scoring on candidate regions.

The main takeaway for Harish/Vaibhav: temporal comparison is the best first baseline. AnomalyCLIP may be useful later, but it should score candidate regions selected by shelf/price-tag context instead of scanning the full image without context.

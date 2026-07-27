# Temporal Data Overlay Images

This folder contains one overlay image per temporal capture from `temporal_data`.

How to read each overlay:

- The original shelf image is shown underneath with slight dimming.
- Faint colored dots show all parsed `EmptyItem` center points across the 15 temporal images.
- The dot colors match `research_outputs/temporal_empty_analysis/temporal_empty_region_center_map.jpg`.
- Bold boxes and bold dots show the `EmptyItem` regions for that specific image.
- Labels use `image_idx.region_idx`, so `1.2` means image 1, empty region 2.

Color mapping follows the center map sequence:

- 1/11 red
- 2/12 blue
- 3/13 green
- 4/14 purple
- 5/15 orange
- 6 brown
- 7 magenta
- 8 cyan
- 9 dark green
- 10 navy

Important caveat: the original annotations are in shelf-coordinate space, not direct pixel coordinates. These overlays use a consistent global coordinate-to-image projection so the temporal relationship is easier to visualize. They are useful for research intuition and comparing repeated locations over time, but they should not be treated as pixel-perfect ground truth boxes.

Generated files: 15 overlay images.

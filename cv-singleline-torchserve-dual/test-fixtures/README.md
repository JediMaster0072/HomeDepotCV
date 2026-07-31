# Test fixtures

- `detection/`: full shelf images for detector smoke tests
- `segmentation/`: one default strip for quick segmentation checks
- `segmentation-comparison/strips/`: the four regression input strips
- `segmentation-comparison/previous/`: masks saved before dependency cleanup
- `segmentation-comparison/new/`: masks regenerated after dependency cleanup

The previous and new masks currently match exactly. Keep these files in Git so
future packaging or dependency changes can be checked pixel-for-pixel.

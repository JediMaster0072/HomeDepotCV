from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


# Input training/validation data that contains the source images and YOLO-format labels.
DATASET_ROOT = Path("/Users/avinash.patel/Downloads/HomeDepotCV/stratified_output_dataset")

# Where this prototype writes the visual overlay sheet and per-sample CSV results.
OUTPUT_ROOT = Path("/Users/avinash.patel/Downloads/HomeDepotCV/research_outputs/rdc_anchor_detection")

# In the stratified dataset labels, class id 1 corresponds to RDC labels.
RDC_CLASS_ID = 1

# Keep the first pass small enough to inspect manually in one contact sheet.
MAX_SAMPLES = 72


@dataclass
class AnchorCandidate:
    """One dark connected component that may represent an orientation anchor."""

    # Bounding box of the dark component inside the cropped label image.
    x: int
    y: int
    w: int
    h: int

    # Number of dark pixels in the component.
    area: int

    # How dense the component is inside its bounding box. Text strokes are often sparse;
    # filled printed bands or markers tend to have a higher fill ratio.
    fill_ratio: float

    # Width / height. This helps distinguish long dark bands from compact marks.
    aspect_ratio: float

    @property
    def center(self) -> tuple[float, float]:
        """Return the component center so we can assign it to a crop quadrant."""
        return self.x + self.w / 2.0, self.y + self.h / 2.0


def yolo_to_pixels(parts: list[str], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Convert one YOLO label row from normalized center/size values to pixel bounds."""
    _, xc, yc, bw, bh = parts[:5]
    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

    # YOLO stores boxes as normalized center x/y plus normalized width/height.
    # The image crop APIs need absolute top-left and bottom-right pixel corners.
    x1 = max(0, int((xc - bw / 2) * image_width))
    y1 = max(0, int((yc - bh / 2) * image_height))
    x2 = min(image_width, int((xc + bw / 2) * image_width))
    y2 = min(image_height, int((yc + bh / 2) * image_height))
    return x1, y1, x2, y2


def crop_with_padding(image: Image.Image, box: tuple[int, int, int, int], pad_pct: float = 0.15) -> Image.Image:
    """Crop a detected label and include some surrounding context."""
    x1, y1, x2, y2 = box
    width, height = image.size

    # Padding helps preserve anchor markers that sit near the edge of a YOLO box.
    pad_x = max(4, int((x2 - x1) * pad_pct))
    pad_y = max(4, int((y2 - y1) * pad_pct))
    return image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y))).convert("RGB")


def quadrant_for_point(x: float, y: float, width: int, height: int) -> str:
    """Map an anchor center to one of four crop quadrants."""
    if y < height / 2 and x < width / 2:
        return "top_left"
    if y < height / 2 and x >= width / 2:
        return "top_right"
    if y >= height / 2 and x < width / 2:
        return "bottom_left"
    return "bottom_right"


def detect_dark_anchor(crop: Image.Image) -> AnchorCandidate | None:
    """Find a likely dark printed anchor component in an RDC crop."""
    # Convert to grayscale because this first prototype only cares about darkness,
    # not color. Blur reduces noisy single-pixel dark artifacts before thresholding.
    arr = np.array(crop)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark printed anchors should be much darker than the label background.
    # The 18th percentile is an adaptive threshold: each crop chooses its own
    # darkness cutoff instead of relying on one fixed lighting assumption.
    thresh = np.percentile(gray, 18)
    dark = (gray <= thresh).astype(np.uint8) * 255

    # Close nearby dark pixels together so parts of the same printed band/mark
    # become one connected component instead of many tiny pieces.
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    candidates: list[AnchorCandidate] = []
    crop_area = crop.width * crop.height

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        # Reject tiny noise and huge regions that are probably background/shadow.
        if area < 20 or area > crop_area * 0.35:
            continue

        # Reject components too small to be a useful printed orientation cue.
        if w < 5 or h < 4:
            continue

        fill_ratio = float(area) / float(w * h)
        aspect_ratio = float(w) / float(max(1, h))

        # For RDC, the anchor can be a horizontal dark band or chunk;
        # keep broad shapes while rejecting tiny digit strokes when possible.
        if fill_ratio < 0.18:
            continue
        candidates.append(AnchorCandidate(int(x), int(y), int(w), int(h), int(area), fill_ratio, aspect_ratio))

    if not candidates:
        return None

    # Pick the largest/densest candidate. This is intentionally simple for the
    # first experiment; the overlay makes false positives easy to spot manually.
    return max(candidates, key=lambda c: (c.area, c.fill_ratio, max(c.aspect_ratio, 1 / max(c.aspect_ratio, 0.001))))


def collect_rdc_crops() -> list[tuple[Image.Image, dict]]:
    """Load RDC examples from the dataset and crop each labeled RDC region."""
    samples: list[tuple[Image.Image, dict]] = []

    # Use both splits because this is exploratory research, not model training.
    for split in ("train", "val"):
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                for line_index, line in enumerate(label_path.read_text().splitlines()):
                    parts = line.split()

                    # Skip non-RDC classes so this prototype only evaluates one label type.
                    if len(parts) < 5 or int(float(parts[0])) != RDC_CLASS_ID:
                        continue

                    # Crop the RDC label from the full shelf image using its YOLO box.
                    crop = crop_with_padding(image, yolo_to_pixels(parts, width, height))
                    samples.append(
                        (
                            crop,
                            {
                                "split": split,
                                "image": image_path.name,
                                "label_line": line_index,
                            },
                        )
                    )

                    # Stop once the output contact sheet is large enough for a quick review.
                    if len(samples) >= MAX_SAMPLES:
                        return samples
    return samples


def main() -> None:
    """Run the RDC anchor experiment and write review artifacts."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    samples = collect_rdc_crops()
    rows: list[dict] = []

    # Build one contact sheet so a reviewer can quickly check whether the red boxes
    # are finding a real anchor or accidentally selecting text/noise.
    thumb_w, thumb_h = 240, 170
    cols = 6
    tile_h = thumb_h + 55
    sheet = Image.new("RGB", (cols * thumb_w, ((len(samples) + cols - 1) // cols) * tile_h), "white")

    for idx, (crop, meta) in enumerate(samples, start=1):
        # Run the actual heuristic on one cropped RDC label.
        anchor = detect_dark_anchor(crop)
        annotated = crop.copy()
        draw = ImageDraw.Draw(annotated)

        quadrant = ""
        anchor_found = anchor is not None
        if anchor:
            cx, cy = anchor.center
            quadrant = quadrant_for_point(cx, cy, crop.width, crop.height)

            # Red box/circle show the selected dark component.
            draw.rectangle([anchor.x, anchor.y, anchor.x + anchor.w, anchor.y + anchor.h], outline="red", width=3)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill="red")

            # Yellow crosshair divides the crop into quadrants. If RDC has a stable
            # anchor, the quadrant should correlate with upright vs rotated labels.
            draw.line([(crop.width / 2, 0), (crop.width / 2, crop.height)], fill="yellow", width=2)
            draw.line([(0, crop.height / 2), (crop.width, crop.height / 2)], fill="yellow", width=2)

        # Put the annotated crop and key metadata into a fixed-size contact-sheet tile.
        tile = Image.new("RGB", (thumb_w, tile_h), "white")
        annotated.thumbnail((thumb_w, thumb_h))
        tile.paste(annotated, ((thumb_w - annotated.width) // 2, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((4, thumb_h + 2), f"{idx:02d} anchor={anchor_found}", fill="black")
        tile_draw.text((4, thumb_h + 18), f"quad={quadrant or 'none'}", fill="black")
        tile_draw.text((4, thumb_h + 34), meta["image"][:34], fill="black")
        sheet.paste(tile, (((idx - 1) % cols) * thumb_w, ((idx - 1) // cols) * tile_h))

        # The CSV stores the same result numerically so we can later sort/filter
        # by quadrant, component size, fill ratio, or source image.
        rows.append(
            {
                "sample_index": idx,
                "anchor_found": anchor_found,
                "quadrant": quadrant,
                "anchor_x": anchor.x if anchor else "",
                "anchor_y": anchor.y if anchor else "",
                "anchor_w": anchor.w if anchor else "",
                "anchor_h": anchor.h if anchor else "",
                "anchor_area": anchor.area if anchor else "",
                "fill_ratio": f"{anchor.fill_ratio:.3f}" if anchor else "",
                "aspect_ratio": f"{anchor.aspect_ratio:.3f}" if anchor else "",
                **meta,
            }
        )

    # Visual artifact for manual review.
    overlay_path = OUTPUT_ROOT / "rdc_anchor_detection_overlay.jpg"
    sheet.save(overlay_path, quality=90)

    # Structured artifact for spreadsheet review and later analysis.
    csv_path = OUTPUT_ROOT / "rdc_anchor_detection_results.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "sample_index",
            "anchor_found",
            "quadrant",
            "anchor_x",
            "anchor_y",
            "anchor_w",
            "anchor_h",
            "anchor_area",
            "fill_ratio",
            "aspect_ratio",
            "split",
            "image",
            "label_line",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"samples={len(samples)}")
    print(f"overlay={overlay_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()

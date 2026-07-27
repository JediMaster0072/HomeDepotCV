from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import numpy as np
from PIL import Image, ImageDraw


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    """Convert a NumPy RGB/grayscale image into a PIL RGB image for drawing."""
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(image[:, :, :3]).convert("RGB")


def _bbox_tuple(bbox: Any) -> tuple[int, int, int, int] | None:
    """Read a BoundingBox-like object without requiring a hard dependency on its class."""
    if bbox is None:
        return None
    return int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)


def _draw_box(draw: ImageDraw.ImageDraw, bbox: Any, color: str, label: str = "") -> None:
    coords = _bbox_tuple(bbox)
    if coords is None:
        return
    x1, y1, x2, y2 = coords
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    if label:
        draw.rectangle([x1, max(0, y1 - 14), x1 + max(70, len(label) * 7), y1], fill="white")
        draw.text((x1 + 2, max(0, y1 - 13)), label, fill=color)


def _resize_keep_aspect(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((max_width, max_height))
    return image


def _make_tile(image: Image.Image, title: str, width: int = 320, height: int = 260) -> Image.Image:
    tile = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((8, 6), title[:60], fill="black")
    body = _resize_keep_aspect(image, width - 16, height - 34)
    tile.paste(body, ((width - body.width) // 2, 28))
    return tile


def _paste_tiles(tiles: list[Image.Image], cols: int = 3, gap: int = 12) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (320, 80), "white")

    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, rows * tile_h + (rows + 1) * gap), "white")

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + gap)
        sheet.paste(tile, (x, y))

    return sheet


def save_validation_contact_sheet(
    raw_image: np.ndarray,
    tracks: Iterable[Any],
    output_path: str | Path,
    groups: list | None = None,
    masked_original: np.ndarray | None = None,
    ocr_results: Iterable[Any] | None = None,
) -> Path:
    """Save one debug image showing detections, crops, segmentation clips, and OCR output.

    This is intentionally a standalone utility so it can be added behind a debug flag
    without changing the production pipeline behavior.

    Suggested usage after `HomeDepotInferencePipeline.run_image()` has built the
    relevant objects:

    - `raw_image`: original RGB image.
    - `tracks`: `ClipTrack` objects.
    - `groups`: strip groups containing optional `masked_strip_clip` entries.
    - `masked_original`: final image sent to Google OCR.
    - `ocr_results`: final `OCRWordResult` objects.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = list(tracks)
    ocr_results = list(ocr_results or [])
    tiles: list[Image.Image] = []

    original = _to_pil_rgb(raw_image)
    original_draw = ImageDraw.Draw(original)
    for track in tracks:
        label = f"{track.det_id}:{track.class_name}"
        _draw_box(original_draw, getattr(track, "orig_bbox", None), "yellow", label)
        _draw_box(original_draw, getattr(track, "buffered_bbox", None), "red")

    for result in ocr_results:
        label = f"OCR {getattr(result, 'text', '')}"
        _draw_box(original_draw, getattr(result, "original_bbox", None), "lime", label)

    tiles.append(_make_tile(original, "Original: yellow=OD, red=buffered, green=OCR", width=520, height=360))

    for track in tracks:
        bbox = _bbox_tuple(getattr(track, "buffered_bbox", None))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        crop = raw_image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        crop_img = _to_pil_rgb(crop)
        title = f"Crop {track.det_id}: {track.class_name} seg={track.seg_found} ocr={track.ocr_found}"
        tiles.append(_make_tile(crop_img, title))

    if groups:
        for group_idx, group in enumerate(groups):
            for item_idx, item in enumerate(group):
                masked_clip = item.get("masked_strip_clip")
                if masked_clip is None or getattr(masked_clip, "size", 0) == 0:
                    continue
                track = item.get("track")
                det_id = getattr(track, "det_id", item_idx)
                clip_img = _to_pil_rgb(masked_clip)
                tiles.append(_make_tile(clip_img, f"Masked seg clip group={group_idx} det={det_id}"))

    if masked_original is not None and masked_original.size > 0:
        tiles.append(_make_tile(_to_pil_rgb(masked_original), "Masked original sent to Google OCR", width=520, height=360))

    sheet = _paste_tiles(tiles, cols=2)
    sheet.save(output_path, quality=92)
    return output_path

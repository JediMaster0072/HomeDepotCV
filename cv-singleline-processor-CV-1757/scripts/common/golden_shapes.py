"""Shared golden-dataset polygon / bbox helpers."""

from __future__ import annotations

SKU_LABEL_SUFFIX = "_SKU"


def shape_points(shape: dict) -> list[tuple[int, int]]:
    points = []
    for point in shape.get("points", []):
        points.append((int(point.get("0", 0)), int(point.get("1", 0))))
    return points


def bbox_from_points(
    points: list[tuple[int, int]],
    image_width: int,
    image_height: int,
    pad_px: int = 8,
) -> tuple[int, int, int, int] | None:
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(xs) - pad_px)
    y1 = max(0, min(ys) - pad_px)
    x2 = min(image_width, max(xs) + pad_px)
    y2 = min(image_height, max(ys) + pad_px)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    *,
    pad_px: int = 0,
    scale: float = 1.0,
) -> tuple[int, int, int, int]:
    """Expand a bbox around its center for full-res context crops."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_w = width * scale / 2.0 + pad_px
    half_h = height * scale / 2.0 + pad_px
    new_x1 = max(0, int(round(center_x - half_w)))
    new_y1 = max(0, int(round(center_y - half_h)))
    new_x2 = min(image_width, int(round(center_x + half_w)))
    new_y2 = min(image_height, int(round(center_y + half_h)))

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return bbox
    return new_x1, new_y1, new_x2, new_y2


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter)


def region_key(image: str, label: str, bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"{image}|{label}|{x1}|{y1}|{x2}|{y2}"


def crop_filename(shape_idx: int, label: str, bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"{shape_idx:03d}_{label}_{x1}_{y1}_{x2}_{y2}.jpg"

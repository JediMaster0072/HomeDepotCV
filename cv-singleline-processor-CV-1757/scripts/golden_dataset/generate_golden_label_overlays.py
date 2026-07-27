import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from common.paths import setup_script_paths

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)
DEFAULT_ORIGINAL_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval"
DEFAULT_EXPECTED_SKU_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"
DEFAULT_DATASET = DEFAULT_EXPECTED_SKU_DATASET if DEFAULT_EXPECTED_SKU_DATASET.exists() else DEFAULT_ORIGINAL_DATASET
DEFAULT_OUTPUT = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "label_overlays_expected_sku"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LABEL_COLORS = {
    "Pallet": (45, 40, 65),
    "Pallet_SKU": (247, 212, 166),
    "RDC": (72, 95, 212),
    "RDC_SKU": (128, 157, 163),
    "Printed_on_Box": (91, 143, 255),
    "Printed_on_Box_SKU": (158, 59, 107),
    "Handwritten": (175, 4, 115),
    "Handwritte_SKU": (22, 22, 217),
    "Handwritten_SKU": (22, 22, 217),
    "Other": (255, 253, 69),
    "Other_SKU": (7, 150, 224),
    "Multiline_Label": (60, 73, 83),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render golden dataset polygons and labels directly onto review images.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Folder containing paired image/json files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Folder for annotated overlay images.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to render.")
    parser.add_argument("--scale", type=float, default=0.5, help="Output scale factor for easier viewing.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Polygon fill alpha.")
    parser.add_argument("--sku-only", action="store_true", help="Render only *_SKU polygons.")
    return parser.parse_args()


def iter_image_json_pairs(dataset_dir: Path):
    for image_path in sorted(dataset_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        json_path = image_path.with_suffix(".json")

        if json_path.exists():
            yield image_path, json_path


def shape_points(shape: dict) -> np.ndarray | None:
    points = []

    for point in shape.get("points", []):
        points.append([int(point.get("0", 0)), int(point.get("1", 0))])

    if not points:
        return None

    return np.array(points, dtype=np.int32)


def label_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label, (255, 255, 255))


def text_origin(points: np.ndarray, image_shape: tuple[int, ...]) -> tuple[int, int]:
    img_h, img_w = image_shape[:2]
    x = int(points[:, 0].min())
    y = int(points[:, 1].min()) - 8

    return max(0, min(x, img_w - 1)), max(18, min(y, img_h - 1))


def draw_label_box(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x2 = min(image.shape[1] - 1, x + text_w + 8)
    y1 = max(0, y - text_h - baseline - 8)
    y2 = min(image.shape[0] - 1, y + baseline + 4)

    cv2.rectangle(image, (x, y1), (x2, y2), color, -1)
    cv2.putText(image, text, (x + 4, y - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_legend(image: np.ndarray):
    x = 20
    y = 35
    row_h = 32

    for idx, (label, color) in enumerate(LABEL_COLORS.items()):
        if label == "Handwritten_SKU":
            continue

        row_y = y + idx * row_h
        cv2.rectangle(image, (x, row_y - 20), (x + 24, row_y + 4), color, -1)
        cv2.putText(
            image,
            label,
            (x + 34, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def shape_text(shape_idx: int, shape: dict) -> str:
    label = shape.get("label", "")
    text = f"{shape_idx}:{label}"
    expected_sku = str(shape.get("expected_sku", "")).strip()

    if expected_sku:
        text = f"{text}={expected_sku}"

    return text


def render_overlay(image_path: Path, json_path: Path, output_path: Path, scale: float, alpha: float, sku_only: bool):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotation = json.loads(json_path.read_text(encoding="utf-8"))
    overlay = image.copy()

    for shape_idx, shape in enumerate(annotation.get("shapes", [])):
        label = shape.get("label", "")

        if sku_only and not label.endswith("_SKU"):
            continue

        points = shape_points(shape)

        if points is None:
            continue

        color = label_color(label)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=4)
        draw_label_box(image, shape_text(shape_idx, shape), text_origin(points, image.shape), color)

    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    draw_legend(image)

    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(iter_image_json_pairs(args.dataset))

    if args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        raise ValueError(f"No image/json pairs found in {args.dataset}")

    for image_path, json_path in pairs:
        output_path = args.output_dir / f"{image_path.stem}_labels.jpg"
        render_overlay(
            image_path=image_path,
            json_path=json_path,
            output_path=output_path,
            scale=args.scale,
            alpha=args.alpha,
            sku_only=args.sku_only,
        )

    print(f"Wrote {len(pairs)} label overlay images to {args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())

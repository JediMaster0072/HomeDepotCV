import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths

SERVICE_ROOT, PROJECT_ROOT, _, _ = setup_script_paths(__file__)

from common.crop_preprocess import (
    DEFAULT_MIN_CROP_SHORT_SIDE,
    angle_slug,
    ensure_min_crop_resolution,
    rotate_image_keep_bounds,
)

import cv2
import numpy as np

from pipeline.ocr_stage import run_google_ocr_words, run_google_ocr_words_with_api_key
from utils.validation_visualizer import save_validation_contact_sheet

from common.sku_review import expected_sku_notes_from_shape, is_na_expected_sku, is_reviewed_expected_sku


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_ORIGINAL_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval"
DEFAULT_EXPECTED_SKU_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"
DEFAULT_DATASET = DEFAULT_EXPECTED_SKU_DATASET if DEFAULT_EXPECTED_SKU_DATASET.exists() else DEFAULT_ORIGINAL_DATASET
DEFAULT_OUTPUT = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests"
DEFAULT_ENV_FILE_CANDIDATES = (
    PROJECT_ROOT / ".env",
    Path.home() / ".home_depot_cv.env",
)
SKU_LABEL_SUFFIX = "_SKU"


def resolve_env_file(env_file: Path | None) -> Path | None:
    if env_file is not None:
        return env_file

    for candidate in DEFAULT_ENV_FILE_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local inventory, crop, OCR, or full-pipeline checks on the golden overhead dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Folder containing paired image/json files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Folder where local test outputs are written.")
    parser.add_argument(
        "--mode",
        choices=("inventory", "crops", "ocr-crops", "rotation-crops", "ocr-rotation-crops", "pipeline"),
        default="inventory",
        help=(
            "inventory: summarize labels only; crops: save annotated SKU crops; "
            "ocr-crops: run Google OCR on annotated SKU crops; rotation-crops: save rotated crop variants; "
            "ocr-rotation-crops: OCR rotated variants; pipeline: run full single-line pipeline."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to process.")
    parser.add_argument("--save-crops", action="store_true", help="Save crop images in ocr-crops mode too.")
    parser.add_argument("--crop-pad-px", type=int, default=8, help="Pixel padding around annotation polygons when cropping.")
    parser.add_argument(
        "--min-crop-short-side",
        type=int,
        default=DEFAULT_MIN_CROP_SHORT_SIDE,
        help="Upscale saved/OCR crops so the short side is at least this many pixels (0 disables).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Optional local .env file containing OCR credentials. "
            "If omitted, the script auto-detects ../.env then ~/.home_depot_cv.env."
        ),
    )
    parser.add_argument(
        "--ocr-auth",
        choices=("auto", "adc", "api-key"),
        default="auto",
        help="OCR auth mode. auto uses GOOGLE_OCR_API_KEY if present, otherwise ADC.",
    )
    parser.add_argument(
        "--google-ocr-api-key-env",
        default="GOOGLE_OCR_API_KEY",
        help="Environment variable name containing the Google OCR API key.",
    )
    parser.add_argument(
        "--rotation-angles",
        default="0,180,-10,10,-5,5",
        help="Comma-separated crop rotation variants for rotation-crops and ocr-rotation-crops modes.",
    )
    parser.add_argument("--store-number", default="", help="Optional store number passed into pipeline mode.")
    parser.add_argument("--debug-validation", action="store_true", help="Write validation contact sheets in pipeline mode.")
    return parser.parse_args()


def load_env_file(env_file: Path | None):
    if env_file is None:
        return

    if not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")

    with env_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def build_ocr_runner(args):
    env_file = resolve_env_file(args.env_file)
    load_env_file(env_file)

    if env_file is not None:
        print(f"Using env file: {env_file}")

    api_key = os.environ.get(args.google_ocr_api_key_env, "")

    if args.ocr_auth in ("auto", "api-key") and api_key:
        return lambda image: run_google_ocr_words_with_api_key(api_key, image)

    if args.ocr_auth == "api-key":
        raise ValueError(f"OCR auth mode is api-key, but {args.google_ocr_api_key_env} is not set")

    from google.cloud import vision

    ocr_client = vision.ImageAnnotatorClient()
    return lambda image: run_google_ocr_words(ocr_client, image)


def iter_image_json_pairs(dataset_dir: Path):
    for image_path in sorted(dataset_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        json_path = image_path.with_suffix(".json")

        if json_path.exists():
            yield image_path, json_path


def load_annotation(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shape_points(shape: dict) -> list[tuple[int, int]]:
    points = []

    for point in shape.get("points", []):
        points.append((int(point.get("0", 0)), int(point.get("1", 0))))

    return points


def normalize_sku_text(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def expected_sku_fields(shape: dict, ocr_results: list[dict]) -> dict[str, str]:
    raw_expected = str(shape.get("expected_sku", "")).strip()
    notes = expected_sku_notes_from_shape(shape)

    if is_na_expected_sku(raw_expected):
        return {
            "expected_sku": "N/A",
            "expected_sku_notes": notes,
            "accuracy_status": "not_applicable",
            "ocr_match": "n/a",
        }

    expected_sku = normalize_sku_text(raw_expected)
    ocr_texts = [normalize_sku_text(result.get("text", "")) for result in ocr_results]
    ocr_texts = [text for text in ocr_texts if text]

    if not expected_sku:
        return {
            "expected_sku": "",
            "expected_sku_notes": notes,
            "accuracy_status": "needs_ground_truth",
            "ocr_match": "",
        }

    matched = expected_sku in ocr_texts

    return {
        "expected_sku": expected_sku,
        "expected_sku_notes": notes,
        "accuracy_status": "correct" if matched else "incorrect",
        "ocr_match": "yes" if matched else "no",
    }


def print_accuracy_summary(rows: list[dict]):
    reviewed = [row for row in rows if is_reviewed_expected_sku(row.get("expected_sku", ""))]
    scorable = [row for row in reviewed if row.get("accuracy_status") not in ("not_applicable",)]
    not_applicable = sum(1 for row in reviewed if row.get("accuracy_status") == "not_applicable")
    missing = len(rows) - len(reviewed)

    if not scorable:
        print(
            f"Accuracy summary: 0 scorable rows, {not_applicable} marked N/A, "
            f"{missing} rows still need expected_sku ground truth"
        )
        return

    correct = sum(1 for row in scorable if row.get("accuracy_status") == "correct")
    incorrect = len(scorable) - correct
    accuracy = correct / len(scorable)
    print(
        "Accuracy summary: "
        f"{correct}/{len(scorable)} correct ({accuracy:.1%}), "
        f"{incorrect} incorrect, {not_applicable} N/A (see expected_sku_notes), "
        f"{missing} rows still need expected_sku"
    )


def bbox_from_points(points: list[tuple[int, int]], image_shape, pad_px: int = 0) -> tuple[int, int, int, int] | None:
    if not points:
        return None

    img_h, img_w = image_shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(xs) - pad_px)
    y1 = max(0, min(ys) - pad_px)
    x2 = min(img_w, max(xs) + pad_px)
    y2 = min(img_h, max(ys) + pad_px)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def parse_rotation_angles(angle_text: str) -> list[float]:
    angles = []

    for item in angle_text.split(","):
        item = item.strip()

        if not item:
            continue

        angles.append(float(item))

    return angles


def label_counts(shapes: list[dict]) -> dict[str, int]:
    counts = {}

    for shape in shapes:
        label = shape.get("label", "")
        counts[label] = counts.get(label, 0) + 1

    return counts


def write_inventory_row(writer, image_path: Path, annotation: dict):
    shapes = annotation.get("shapes", [])
    counts = label_counts(shapes)
    sku_count = sum(count for label, count in counts.items() if label.endswith(SKU_LABEL_SUFFIX))

    writer.writerow(
        {
            "image": image_path.name,
            "image_width": annotation.get("imageWidth", ""),
            "image_height": annotation.get("imageHeight", ""),
            "shape_count": len(shapes),
            "sku_region_count": sku_count,
            "labels": json.dumps(counts, sort_keys=True),
        }
    )


def save_annotation_crops(
    image_path: Path,
    annotation: dict,
    output_dir: Path,
    crop_pad_px: int,
    ocr_runner=None,
    save_crops: bool = True,
    min_crop_short_side: int = DEFAULT_MIN_CROP_SHORT_SIDE,
):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = image_bgr[:, :, ::-1]
    rows = []
    crops_dir = output_dir / "crops" / image_path.stem

    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for shape_idx, shape in enumerate(annotation.get("shapes", [])):
        label = shape.get("label", "")

        if not label.endswith(SKU_LABEL_SUFFIX):
            continue

        points = shape_points(shape)
        bbox = bbox_from_points(points, image_rgb.shape, pad_px=crop_pad_px)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        crop_rgb = image_rgb[y1:y2, x1:x2]
        crop_rgb = ensure_min_crop_resolution(crop_rgb, min_short_side=min_crop_short_side)
        crop_name = f"{shape_idx:03d}_{label}_{x1}_{y1}_{x2}_{y2}.jpg"
        crop_path = crops_dir / crop_name

        if save_crops:
            cv2.imwrite(str(crop_path), crop_rgb[:, :, ::-1])

        ocr_results = []

        if ocr_runner is not None:
            ocr_results = ocr_runner(crop_rgb)

        row = {
            "image": image_path.name,
            "shape_idx": shape_idx,
            "label": label,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "crop_path": str(crop_path) if save_crops else "",
            "ocr_count": len(ocr_results),
            "ocr_texts": "|".join(result["text"] for result in ocr_results),
            "ocr_sources": "|".join(result.get("source", "") for result in ocr_results),
        }
        row.update(expected_sku_fields(shape, ocr_results))
        rows.append(row)

    return rows


def save_rotation_crop_variants(
    image_path: Path,
    annotation: dict,
    output_dir: Path,
    crop_pad_px: int,
    rotation_angles: list[float],
    ocr_runner=None,
    min_crop_short_side: int = DEFAULT_MIN_CROP_SHORT_SIDE,
):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = image_bgr[:, :, ::-1]
    rows = []
    rotations_dir = output_dir / "rotation_crops" / image_path.stem
    rotations_dir.mkdir(parents=True, exist_ok=True)
    for shape_idx, shape in enumerate(annotation.get("shapes", [])):
        label = shape.get("label", "")

        if not label.endswith(SKU_LABEL_SUFFIX):
            continue

        points = shape_points(shape)
        bbox = bbox_from_points(points, image_rgb.shape, pad_px=crop_pad_px)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        crop_rgb = ensure_min_crop_resolution(
            image_rgb[y1:y2, x1:x2],
            min_short_side=min_crop_short_side,
        )

        for angle in rotation_angles:
            rotated = rotate_image_keep_bounds(crop_rgb, angle)
            variant_name = f"{shape_idx:03d}_{label}_{x1}_{y1}_{x2}_{y2}_rot{angle_slug(angle)}.jpg"
            variant_path = rotations_dir / variant_name
            cv2.imwrite(str(variant_path), rotated[:, :, ::-1])

            ocr_results = []

            if ocr_runner is not None:
                ocr_results = ocr_runner(rotated)

            row = {
                "image": image_path.name,
                "shape_idx": shape_idx,
                "label": label,
                "rotation_angle": angle,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "variant_path": str(variant_path),
                "ocr_count": len(ocr_results),
                "ocr_texts": "|".join(result["text"] for result in ocr_results),
                "ocr_sources": "|".join(result.get("source", "") for result in ocr_results),
            }
            row.update(expected_sku_fields(shape, ocr_results))
            rows.append(row)

    return rows


def run_pipeline_on_image(image_path: Path, output_dir: Path, store_number: str, debug_validation: bool):
    from new_inference_pipeline_full_image import HomeDepotInferencePipeline

    if debug_validation:
        os.environ["CV_SINGLELINE_DEBUG_VALIDATION"] = "true"

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pipeline = HomeDepotInferencePipeline(debug_output_dir=output_dir / "pipeline_debug")
    status, result = pipeline.run_image(image_bgr[:, :, ::-1], file_name=str(image_path), store_number=store_number)

    row = {
        "image": image_path.name,
        "status": status,
        "ocr_count": 0,
        "ocr_texts": "",
        "ocr_sources": "",
        "ocr_enhanced_retry_used": "",
        "ocr_rot180_retry_used": "",
        "num_ocr_calls": "",
    }

    if result is None:
        return row

    row.update(
        {
            "ocr_count": len(result.raw_ocr_results),
            "ocr_texts": "|".join(item.text for item in result.raw_ocr_results),
            "ocr_sources": "|".join(item.source for item in result.raw_ocr_results),
            "ocr_enhanced_retry_used": result.metadata.get("ocr_enhanced_retry_used", ""),
            "ocr_rot180_retry_used": result.metadata.get("ocr_rot180_retry_used", ""),
            "num_ocr_calls": result.metadata.get("num_ocr_calls", ""),
        }
    )

    if debug_validation:
        tracks = result.metadata.get("tracked_clips", [])
        save_validation_contact_sheet(
            raw_image=image_bgr[:, :, ::-1],
            tracks=tracks,
            output_path=output_dir / "pipeline_debug" / f"{image_path.stem}_contact_sheet.jpg",
            ocr_results=result.raw_ocr_results,
        )

    return row


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(iter_image_json_pairs(args.dataset))

    if args.limit > 0:
        pairs = pairs[: args.limit]

    if args.mode == "inventory":
        output_csv = args.output_dir / "inventory.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image", "image_width", "image_height", "shape_count", "sku_region_count", "labels"],
            )
            writer.writeheader()

            for image_path, json_path in pairs:
                write_inventory_row(writer, image_path, load_annotation(json_path))

        print(f"Wrote inventory: {output_csv}")
        return

    if args.mode in ("crops", "ocr-crops"):
        output_csv = args.output_dir / f"{args.mode}_summary.csv"
        rows = []
        ocr_runner = build_ocr_runner(args) if args.mode == "ocr-crops" else None

        for image_path, json_path in pairs:
            rows.extend(
                save_annotation_crops(
                    image_path=image_path,
                    annotation=load_annotation(json_path),
                    output_dir=args.output_dir,
                    crop_pad_px=args.crop_pad_px,
                    ocr_runner=ocr_runner,
                    save_crops=args.save_crops or args.mode == "crops",
                    min_crop_short_side=args.min_crop_short_side,
                )
            )

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image",
                "shape_idx",
                "label",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "crop_path",
                "ocr_count",
                "ocr_texts",
                "ocr_sources",
                "expected_sku",
                "expected_sku_notes",
                "accuracy_status",
                "ocr_match",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote {args.mode} summary: {output_csv}")
        print_accuracy_summary(rows)
        return

    if args.mode in ("rotation-crops", "ocr-rotation-crops"):
        output_csv = args.output_dir / f"{args.mode}_summary.csv"
        rotation_angles = parse_rotation_angles(args.rotation_angles)
        rows = []
        ocr_runner = build_ocr_runner(args) if args.mode == "ocr-rotation-crops" else None
        total_images = len(pairs)
        angles_per_crop = len(rotation_angles)

        if args.mode == "ocr-rotation-crops":
            est_calls = total_images * angles_per_crop
            print(
                f"OCR rotation mode: {total_images} images × {angles_per_crop} angles "
                f"≈ {est_calls} Google OCR calls (no progress file until complete; this can take hours).",
                flush=True,
            )

        for image_idx, (image_path, json_path) in enumerate(pairs, start=1):
            if args.mode == "ocr-rotation-crops":
                print(f"[{image_idx}/{total_images}] {image_path.name}", flush=True)
            rows.extend(
                save_rotation_crop_variants(
                    image_path=image_path,
                    annotation=load_annotation(json_path),
                    output_dir=args.output_dir,
                    crop_pad_px=args.crop_pad_px,
                    rotation_angles=rotation_angles,
                    ocr_runner=ocr_runner,
                    min_crop_short_side=args.min_crop_short_side,
                )
            )

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image",
                "shape_idx",
                "label",
                "rotation_angle",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "variant_path",
                "ocr_count",
                "ocr_texts",
                "ocr_sources",
                "expected_sku",
                "expected_sku_notes",
                "accuracy_status",
                "ocr_match",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote {args.mode} summary: {output_csv}")
        print_accuracy_summary(rows)
        return

    if args.mode == "pipeline":
        output_csv = args.output_dir / "pipeline_summary.csv"
        rows = []

        for image_path, _json_path in pairs:
            rows.append(run_pipeline_on_image(image_path, args.output_dir, args.store_number, args.debug_validation))

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image",
                "status",
                "ocr_count",
                "ocr_texts",
                "ocr_sources",
                "ocr_enhanced_retry_used",
                "ocr_rot180_retry_used",
                "num_ocr_calls",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote pipeline summary: {output_csv}")


if __name__ == "__main__":
    main()

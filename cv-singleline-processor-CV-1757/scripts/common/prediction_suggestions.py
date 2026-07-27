"""Map pipeline predictions.json detections onto golden-dataset annotation rows."""

from __future__ import annotations

import json
import re
from pathlib import Path

from common.sku_review import parse_expected_sku_input


def _bbox(value: dict | None) -> tuple[int, int, int, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        bbox = tuple(int(value[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _row_bbox(row: dict) -> tuple[int, int, int, int] | None:
    try:
        bbox = tuple(
            int(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        )
    except (KeyError, TypeError, ValueError):
        return None
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _class_token(value: str) -> str:
    token = re.sub(r"[^a-z]", "", str(value or "").lower())
    token = token.removesuffix("sku")
    if token in {"handwritte", "handwritten"}:
        return "handwritten"
    return token


def _classes_match(prediction: dict, row: dict) -> bool:
    predicted = _class_token(prediction.get("class_name", ""))
    annotated = _class_token(row.get("label", ""))
    return not predicted or not annotated or predicted == annotated


def _prediction_text(prediction: dict) -> str:
    candidates: list[str] = []
    for word in prediction.get("ocr_words") or []:
        value, error = parse_expected_sku_input(
            str(word.get("text", "") or word.get("raw_text", ""))
        )
        if not error and value not in {None, "N/A", "X"}:
            candidates.append(value)
    if not candidates:
        return ""
    return max(candidates, key=lambda value: (len(value), value))


def prediction_text(prediction: dict) -> str:
    """Return the preferred normalized 6- or 10-digit OCR value for one track."""
    return _prediction_text(prediction)


def match_prediction_to_row(
    prediction: dict,
    rows: list[dict],
    minimum_iou: float = 0.0,
) -> tuple[dict | None, float]:
    """Match one prediction to the same-class ground-truth row with highest IoU."""
    prediction_bbox = _bbox(prediction.get("orig_bbox"))
    if prediction_bbox is None:
        return None, 0.0
    matches = [
        (_iou(prediction_bbox, row_bbox), row)
        for row in rows
        if _classes_match(prediction, row)
        and (row_bbox := _row_bbox(row)) is not None
    ]
    if not matches:
        return None, 0.0
    overlap, row = max(matches, key=lambda item: item[0])
    return (row, overlap) if overlap >= minimum_iou else (None, overlap)


def load_prediction_tracks(
    predictions_path: Path,
    rows: list[dict],
) -> dict[str, list[dict]]:
    """Load either one predictions JSON or a directory of per-image JSON files."""
    if not predictions_path.exists():
        return {}
    if predictions_path.is_dir():
        grouped: dict[str, list[dict]] = {}
        for json_path in sorted(predictions_path.glob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                grouped[f"{json_path.stem}.jpg"] = [
                    item for item in payload if isinstance(item, dict)
                ]
        return grouped

    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        predictions = payload.get("predictions") or payload.get("detections") or []
        global_image = str(
            payload.get("image") or payload.get("image_name") or payload.get("filename") or ""
        ).strip()
    elif isinstance(payload, list):
        predictions = payload
        global_image = ""
    else:
        return {}
    predictions = [item for item in predictions if isinstance(item, dict)]

    grouped: dict[str, list[dict]] = {}
    without_image: list[dict] = []
    for prediction in predictions:
        image = str(
            prediction.get("image")
            or prediction.get("image_name")
            or prediction.get("filename")
            or global_image
            or ""
        ).strip()
        if image:
            grouped.setdefault(Path(image).name, []).append(prediction)
        else:
            without_image.append(prediction)
    if without_image:
        inferred_image = _infer_image(without_image, rows)
        if inferred_image:
            grouped.setdefault(inferred_image, []).extend(without_image)
    return grouped


def available_prediction_images(predictions_path: Path) -> list[str]:
    if predictions_path.is_dir():
        return sorted(f"{path.stem}.jpg" for path in predictions_path.glob("*.json"))
    return []


def load_prediction_tracks_for_image(
    predictions_path: Path,
    image_name: str,
    rows: list[dict],
) -> list[dict]:
    if predictions_path.is_dir():
        json_path = predictions_path / f"{Path(image_name).stem}.json"
        if not json_path.exists():
            return []
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    return load_prediction_tracks(predictions_path, rows).get(image_name, [])


def _infer_image(predictions: list[dict], rows: list[dict]) -> str:
    rows_by_image: dict[str, list[dict]] = {}
    for row in rows:
        image = str(row.get("image", "") or "").strip()
        if image:
            rows_by_image.setdefault(image, []).append(row)

    image_scores: list[tuple[float, str]] = []
    for image, image_rows in rows_by_image.items():
        score = 0.0
        for prediction in predictions:
            prediction_bbox = _bbox(prediction.get("orig_bbox"))
            if prediction_bbox is None:
                continue
            score += max(
                (
                    _iou(prediction_bbox, row_bbox)
                    for row in image_rows
                    if _classes_match(prediction, row)
                    and (row_bbox := _row_bbox(row)) is not None
                ),
                default=0.0,
            )
        image_scores.append((score, image))

    if not image_scores:
        return ""
    best_score, best_image = max(image_scores)
    return best_image if best_score >= 1.0 else ""


def prediction_suggestions_for_rows(
    predictions_path: Path,
    rows: list[dict],
) -> tuple[set[str], dict[str, dict]]:
    """
    Return images represented by predictions.json and suggestions keyed by region_key.

    Preferred payloads include an image filename. A legacy bare list, such as the
    current predictions.json, is associated to one image by aggregate bbox overlap.
    """
    if not predictions_path.exists():
        return set(), {}

    rows_by_image: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_image.setdefault(str(row.get("image", "")), []).append(row)

    if predictions_path.is_dir():
        json_paths = sorted(predictions_path.glob("*.json"))
        represented_images = {f"{path.stem}.jpg" for path in json_paths}

        def iter_grouped_items():
            for json_path in json_paths:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                yield (
                    f"{json_path.stem}.jpg",
                    [item for item in payload if isinstance(item, dict)]
                    if isinstance(payload, list)
                    else [],
                )

        grouped_items = iter_grouped_items()
    else:
        grouped = load_prediction_tracks(predictions_path, rows)
        represented_images = set(grouped)
        grouped_items = grouped.items()

    suggestions: dict[str, dict] = {}
    for image, image_predictions in grouped_items:
        image_rows = rows_by_image.get(image, [])
        for prediction in image_predictions:
            text = _prediction_text(prediction)
            if not text:
                continue
            row, overlap = match_prediction_to_row(
                prediction,
                image_rows,
                minimum_iou=0.25,
            )
            if row is None:
                continue
            key = str(row.get("region_key", "") or "").strip()
            if not key:
                continue
            candidate = {
                "text": text,
                "det_id": prediction.get("det_id"),
                "confidence": prediction.get("confidence"),
                "overlap": overlap,
            }
            existing = suggestions.get(key)
            if existing is None or float(candidate["confidence"] or 0) > float(
                existing["confidence"] or 0
            ):
                suggestions[key] = candidate
    return represented_images, suggestions

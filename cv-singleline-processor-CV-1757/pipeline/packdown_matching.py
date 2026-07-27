"""Join selling-area missing SKUs to storewide inventory observations."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable


def normalize_packdown_sku(value: Any) -> str | None:
    """Normalize a validated 6- or 10-digit SKU to the published 10-digit form."""
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 6:
        return digits.zfill(10)
    if len(digits) == 10:
        return digits
    return None


def _parse_legacy_boxes(value: Any) -> list[list[float]]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))
    boxes = []
    for offset in range(0, len(numbers), 4):
        coords = numbers[offset:offset + 4]
        if len(coords) == 4:
            boxes.append([float(coord) for coord in coords])
    return boxes


def inventory_observations_from_payload(payload: dict) -> list[dict]:
    """
    Convert one single-line or multiline OCR payload into inventory observations.

    The aisle/bay is where the box was observed. It is not assumed to be the
    selling location for that SKU.
    """
    if not isinstance(payload, dict):
        return []

    common = {
        "store_number": str(payload.get("store_number") or "").zfill(4),
        "inventory_aisle": str(payload.get("aisle_number") or "").zfill(2),
        "inventory_bay": str(payload.get("bay_number") or "").zfill(3),
        "photo_timestamp": payload.get("photo_timestamp"),
        "photo_location_path": payload.get("photo_location_path"),
        "filename": payload.get("filename"),
        "process_source": payload.get("process_source"),
    }

    observations = []
    rich_observations = payload.get("inventory_observations")
    if isinstance(rich_observations, list):
        for raw in rich_observations:
            if not isinstance(raw, dict):
                continue
            sku = normalize_packdown_sku(raw.get("sku"))
            if sku is None:
                continue
            observations.append(
                {
                    **common,
                    **raw,
                    "sku": sku,
                    "inventory_bbox": raw.get("bbox"),
                }
            )
        return observations

    # Compatibility with existing single-line/multiline payloads.
    bounding_boxes = payload.get("bounding_boxes")
    if not isinstance(bounding_boxes, dict):
        return []
    for raw_sku, encoded_boxes in bounding_boxes.items():
        sku = normalize_packdown_sku(raw_sku)
        if sku is None:
            continue
        for bbox in _parse_legacy_boxes(encoded_boxes):
            observations.append(
                {
                    **common,
                    "sku": sku,
                    "inventory_bbox": bbox,
                    "confidence": 1.0,
                    "source": "legacy_inventory_payload",
                }
            )
    return observations


def build_store_inventory_index(payloads: Iterable[dict]) -> dict[str, list[dict]]:
    """Build an exact-SKU index across all observed inventory boxes."""
    index: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        for observation in inventory_observations_from_payload(payload):
            index[observation["sku"]].append(observation)
    return dict(index)


def missing_sku_requests_from_empty_grid(
    empty_grid_info: list[list[tuple]],
    *,
    store_number: str,
    selling_aisle: str,
    selling_bay: str,
) -> list[dict]:
    """Adapt selling-processor empty-grid output into packdown lookup requests."""
    requests = []
    for shelf_index, shelf in enumerate(empty_grid_info or []):
        for position_index, cell in enumerate(shelf):
            if not isinstance(cell, (list, tuple)) or len(cell) < 6:
                continue
            empty_state, raw_sku, image_price, retail_price, location_id, confidence = cell[:6]
            if empty_state != "Empty":
                continue
            sku = normalize_packdown_sku(raw_sku)
            if sku is None:
                continue
            requests.append(
                {
                    "sku": sku,
                    "store_number": str(store_number).zfill(4),
                    "selling_aisle": str(selling_aisle).zfill(2),
                    "selling_bay": str(selling_bay).zfill(3),
                    "selling_location_id": location_id,
                    "selling_shelf_index": shelf_index,
                    "selling_position_index": position_index,
                    "image_price": image_price,
                    "retail_price": retail_price,
                    "confidence": confidence,
                }
            )
    return requests


def build_packdown_candidates(
    missing_sku_requests: Iterable[dict],
    inventory_index: dict[str, list[dict]],
) -> list[dict]:
    """
    Match missing selling SKUs to exact-SKU inventory observations storewide.

    This returns candidates, not an automatic stocking task. Routing, inventory
    availability, and task assignment remain downstream responsibilities.
    """
    results = []
    for request in missing_sku_requests:
        sku = normalize_packdown_sku(request.get("sku"))
        store_number = str(request.get("store_number") or "").zfill(4)
        matches = [
            observation
            for observation in inventory_index.get(sku or "", [])
            if observation.get("store_number") == store_number
        ]
        matches.sort(
            key=lambda item: (
                -float(item.get("confidence", 0.0) or 0.0),
                str(item.get("photo_timestamp") or ""),
            )
        )
        results.append(
            {
                "request": dict(request),
                "status": "INVENTORY_FOUND" if matches else "NO_INVENTORY_MATCH",
                "inventory_matches": matches,
            }
        )
    return results

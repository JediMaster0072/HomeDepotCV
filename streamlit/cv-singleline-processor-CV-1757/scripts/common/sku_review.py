"""Shared helpers for expected_sku human review and evaluation."""

from __future__ import annotations

import re

NA_SKU_VALUES = frozenset({"N/A", "NA", "N-A"})


def normalize_sku_digits(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))


def is_na_expected_sku(value: str) -> bool:
    return str(value or "").strip().upper() in NA_SKU_VALUES


def is_reviewed_expected_sku(value: str) -> bool:
    return bool(normalize_sku_digits(value)) or is_na_expected_sku(value)


def parse_expected_sku_input(raw: str) -> tuple[str | None, str | None]:
    """
    Parse reviewer input.

    Returns:
        (stored_value, error_message). error_message is set when input is invalid.
    """
    text = str(raw or "").strip()
    if not text:
        return None, "expected_sku is required (enter SKU digits or N/A with notes)."

    if is_na_expected_sku(text):
        return "N/A", None

    digits = normalize_sku_digits(text)
    if digits:
        return digits, None

    return None, "expected_sku must be digits or N/A."


def expected_sku_notes_from_shape(shape: dict) -> str:
    return str(
        shape.get("expected_sku_notes", "")
        or shape.get("notes", "")
        or ""
    ).strip()

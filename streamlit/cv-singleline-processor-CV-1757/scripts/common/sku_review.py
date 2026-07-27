"""Shared helpers for expected_sku human review and evaluation."""

from __future__ import annotations

import re

NA_SKU_VALUES = frozenset({"N/A", "NA", "N-A", "X"})
VALID_SKU_LENGTHS = frozenset({6, 10})
_SKU_SEPARATOR_RE = re.compile(r"[\s\-_./]")
_SKU_TOKEN_RE = re.compile(r"^[0-9Xx]+$")


def normalize_sku_digits(value: str) -> str:
    """Digits only — used for OCR exact-match comparisons."""
    return re.sub(r"\D", "", str(value or ""))


def normalize_sku_token(value: str) -> str:
    """Digits + X placeholders (X uppercased). Separators are removed."""
    cleaned = _SKU_SEPARATOR_RE.sub("", str(value or "").strip())
    if not cleaned or not _SKU_TOKEN_RE.fullmatch(cleaned):
        return ""
    return cleaned.upper()


def is_valid_sku_token(value: str) -> bool:
    return len(normalize_sku_token(value)) in VALID_SKU_LENGTHS


def is_na_expected_sku(value: str) -> bool:
    return str(value or "").strip().upper() in NA_SKU_VALUES


def is_reviewed_expected_sku(value: str) -> bool:
    return is_valid_sku_token(value) or is_na_expected_sku(value)


def parse_expected_sku_input(raw: str) -> tuple[str | None, str | None]:
    """
    Parse reviewer input.

    Accepts exactly 6 or 10 characters from digits and optional X placeholders
    (for digits that are not visible). Bare X / N/A still means not visible.

    Returns:
        (stored_value, error_message). error_message is set when input is invalid.
    """
    text = str(raw or "").strip()
    if not text:
        return None, "expected_sku is required (enter SKU digits or N/A with notes)."

    if is_na_expected_sku(text):
        return "X" if text.upper() == "X" else "N/A", None

    token = normalize_sku_token(text)
    if len(token) in VALID_SKU_LENGTHS:
        return token, None

    return (
        None,
        "expected_sku must be exactly 6 or 10 characters (digits and X for unclear digits).",
    )


def expected_sku_notes_from_shape(shape: dict) -> str:
    return str(
        shape.get("expected_sku_notes", "")
        or shape.get("notes", "")
        or ""
    ).strip()

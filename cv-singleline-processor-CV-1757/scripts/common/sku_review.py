"""Shared helpers for expected_sku human review and evaluation."""

from __future__ import annotations

import re

NA_SKU_VALUES = frozenset({"N/A", "NA", "N-A", "X"})
SCORABLE_REVIEW_STATUS = "scorable"
NON_SCORABLE_REVIEW_STATUSES = frozenset(
    {
        "non_scorable",
        "motion_blur",
        "glare",
        "occluded",
        "cropped",
        "resolution_too_low",
        "damaged_label",
        "unreadable",
        "ambiguous",
        "other",
    }
)
LEGACY_SCORABLE_REVIEW_STATUSES = frozenset({"reviewed", "reviewed_ocr_assisted"})
VALID_SKU_LENGTHS = frozenset({6, 10})
# Separators reviewers may type; stripped before length checks.
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
    return cleaned.upper()  # digits unchanged; x -> X


def is_valid_sku_token(value: str) -> bool:
    token = normalize_sku_token(value)
    return len(token) in VALID_SKU_LENGTHS


def is_na_expected_sku(value: str) -> bool:
    return str(value or "").strip().upper() in NA_SKU_VALUES


def is_reviewed_expected_sku(value: str) -> bool:
    return is_valid_sku_token(value) or is_na_expected_sku(value)


def scorability_from_expected_sku(value: str) -> str:
    if is_na_expected_sku(value):
        return "non-scorable"
    if is_valid_sku_token(value):
        return "scorable"
    return ""


def sku_digit_class(value: str) -> str:
    """Classify a reviewed value without relying on a separate manual label."""
    if is_na_expected_sku(value):
        return "not-visible"
    token = normalize_sku_token(value)
    if len(token) == 6:
        return "6-digit"
    if len(token) == 10:
        return "10-digit"
    return ""


def normalize_review_status(
    review_status: str,
    expected_sku: str = "",
    notes: str = "",
) -> str:
    """Return the canonical scorability status, including legacy CSV rows."""
    status = str(review_status or "").strip().lower().replace(" ", "_")
    if status == SCORABLE_REVIEW_STATUS or status in LEGACY_SCORABLE_REVIEW_STATUSES:
        return SCORABLE_REVIEW_STATUS
    if status in NON_SCORABLE_REVIEW_STATUSES:
        return status

    if status == "not_applicable" or is_na_expected_sku(expected_sku):
        note_text = str(notes or "").lower()
        reason_terms = (
            ("motion_blur", ("motion blur", "blur")),
            ("glare", ("glare",)),
            ("occluded", ("occluded", "occlusion")),
            ("cropped", ("cropped", "crop cut")),
            ("resolution_too_low", ("resolution", "too small", "low res")),
            ("damaged_label", ("damaged", "damage")),
            ("ambiguous", ("ambiguous", "uncertain")),
        )
        for reason, terms in reason_terms:
            if any(term in note_text for term in terms):
                return reason
        return "unreadable"

    if is_valid_sku_token(expected_sku):
        return SCORABLE_REVIEW_STATUS
    return ""


def is_scorable_review(
    review_status: str,
    expected_sku: str = "",
    notes: str = "",
) -> bool:
    return normalize_review_status(review_status, expected_sku, notes) == SCORABLE_REVIEW_STATUS


def is_reviewed_annotation(
    review_status: str,
    expected_sku: str = "",
    notes: str = "",
) -> bool:
    return bool(normalize_review_status(review_status, expected_sku, notes))


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

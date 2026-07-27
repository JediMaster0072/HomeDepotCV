import math
import re
from common_config import BoundingBox


# ----------------------------
# Google annotation helpers
# ----------------------------


# Pipeline order: 32.1
# Description: Reads OCR annotation text from either dict-style or protobuf-style Google OCR objects.
def _ann_text(item) -> str:
    """
    Works for both:
    - Google protobuf-style objects: item.description
    - JSON/dict style: item["description"]
    """
    if isinstance(item, dict):
        return (item.get("description") or "").strip()

    return (getattr(item, "description", "") or "").strip()


# Pipeline order: 32.3
# Description: Reads OCR polygon vertices from either dict-style or protobuf-style Google OCR objects.
def _get_vertices(item):
    """
    Works for both:
    - Google protobuf-style objects: item.bounding_poly.vertices
    - JSON/dict style: item["boundingPoly"]["vertices"]
    """
    if isinstance(item, dict):
        poly = item.get("boundingPoly") or item.get("bounding_poly") or {}
        return poly.get("vertices", []) or []

    bounding_poly = getattr(item, "bounding_poly", None)
    if bounding_poly is None:
        return []

    return list(getattr(bounding_poly, "vertices", []) or [])


# Pipeline order: 32.4
# Description: Extracts one coordinate value from an OCR polygon vertex.
def _vertex_value(vertex, key: str) -> int:
    if isinstance(vertex, dict):
        return int(vertex.get(key, 0) or 0)

    return int(getattr(vertex, key, 0) or 0)


# Pipeline order: 32.2
# Description: Converts a Google OCR polygon annotation into a rectangular BoundingBox.
def bbox_from_google_annotation(item) -> BoundingBox | None:
    vertices = _get_vertices(item)

    if len(vertices) < 4:
        return None

    xs = [_vertex_value(v, "x") for v in vertices]
    ys = [_vertex_value(v, "y") for v in vertices]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    if x2 <= x1 or y2 <= y1:
        return None

    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


# Pipeline order: 34.3
# Description: Merges multiple OCR word boxes into one enclosing bounding box for multi-token SKUs.
def merge_bboxes(boxes: list[BoundingBox]) -> BoundingBox:
    if not boxes:
        raise ValueError("Cannot merge empty bbox list")

    return BoundingBox(
        x1=min(b.x1 for b in boxes),
        y1=min(b.y1 for b in boxes),
        x2=max(b.x2 for b in boxes),
        y2=max(b.y2 for b in boxes),
    )


# ----------------------------
# SKU normalization
# ----------------------------


# Pipeline order: 34.2
# Description: Validates and normalizes noisy OCR text into a 6-digit or 10-digit SKU when possible.
def normalize_sku_candidate(text: str) -> str | None:
    """
    Valid SKU rules:
    - SKU must normalize to exactly 6 or 10 digits.
    - Alphabets are not allowed.
    - Internal allowed chars are only:
        digits, spaces, hyphens.
    - Edge symbols like "$" or trailing ":" are allowed only at the edges.
    - Slashes are NOT allowed, so dates like "02/05/26" are rejected.
    """

    if not text:
        return None

    text = text.strip()

    if not text:
        return None

    # Pattern for text with 10 digit SKU numbers
    pattern_10 = re.compile(r"(?<!\d)(\d{10})(?!\d)")
    match = pattern_10.search(text)
    if match:
        if match.group(1).isdigit():
            return match.group(1)

    # Pattern for text with 6 digit SKU numbers
    pattern_6 = re.compile(r"(?<!\d)(\d{6})(?!\d)")
    match = pattern_6.search(text)
    if match:
        if match.group(1).isdigit():
            return match.group(1)

    # Reject anything with alphabets.
    # Example: "D28 04/01/26", "QTY", "TO14-390"
    if re.search(r"[A-Za-z]", text):
        return None

    # Remove edge noise only.
    # Example: "$1008981864" -> "1008981864"
    # Example: "1007343429-" -> "1007343429"
    text = re.sub(r"^[^\d]+", "", text)
    text = re.sub(r"[^\d]+$", "", text)

    if not text:
        return None

    # Important:
    # Do NOT allow slash.
    # Otherwise "023 02/05/26" could become fake digits.
    allowed_pattern = r"^[0-9\s\-\u2010\u2011\u2012\u2013\u2014]+$"
    if not re.fullmatch(allowed_pattern, text):
        return None

    normalized = re.sub(r"[\s\-\u2010\u2011\u2012\u2013\u2014]+", "", text)

    if not normalized.isdigit():
        return None

    # Case 1: 11 characters with "10" at position 1-2, extract 10-digit SKU
    if len(normalized) == 11 and normalized[1:3] == "10":
        return normalized[1:11]

    # Case 3: 11 or 12 digits starting with "10", extract the first 10 digits
    if len(normalized) in [11, 12] and normalized[0:2] == "10":
        return normalized[0:10]

    # Case 4: Length > 10, extract last 10 digits if they start with "10"
    # This handles cases like "22 1004215338" or "-338 1004215338" → "1004215338"
    if len(normalized) > 10:
        last_10 = normalized[-10:]
        if last_10[0:2] == "10":
            return last_10

    if len(normalized) not in {6, 10}:
        return None

    return normalized


# ----------------------------
# OCR record building
# ----------------------------


# Pipeline order: 32
# Description: Converts Google OCR word annotations into records with text, bbox, center, width, and height.
def build_ocr_records(annotations) -> list[dict]:
    """
    annotations[0] = full text block
    annotations[1:] = individual OCR nodes
    """

    records = []

    for idx, item in enumerate(annotations[1:], start=1):
        text = _ann_text(item)

        if not text:
            continue

        bbox = bbox_from_google_annotation(item)

        if bbox is None:
            continue

        records.append(
            {
                "idx": idx,
                "text": text,
                "bbox": bbox,
                "cx": (bbox.x1 + bbox.x2) / 2.0,
                "cy": (bbox.y1 + bbox.y2) / 2.0,
                "h": max(1, bbox.y2 - bbox.y1),
                "w": max(1, bbox.x2 - bbox.x1),
            }
        )

    return records


# Pipeline order: 33.1
# Description: Removes whitespace so full-text OCR lines can be aligned with individual OCR nodes.
def _alignment_key(text: str) -> str:
    """
    Used only for aligning annotations[0].description lines
    with annotations[1:] nodes.

    Example:
    line:  "D28 04/01/26"
    nodes: "D28", "04/01/26"

    Both become:
    "D2804/01/26"
    """
    return re.sub(r"\s+", "", text.strip())


# Pipeline order: 33
# Description: Aligns Google full-text lines with individual OCR records so split SKUs can be reconstructed.
def build_full_text_groups(annotations, records: list[dict]) -> list[dict]:
    """
    Build logical OCR groups using annotations[0].description split by newline.

    This prevents unsafe visual-y merges like:
        "1046652" + "023" -> "1046652023"

    because those are separate newline groups in annotations[0].description.
    """

    if not annotations or not records:
        return []

    full_text = _ann_text(annotations[0])

    # Conservative fallback:
    # If the full text block is missing, do not do risky multi-node merging.
    if not full_text:
        return [
            {
                "group_id": i,
                "group_text": r["text"],
                "records": [r],
            }
            for i, r in enumerate(records)
        ]

    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    groups = []
    record_pos = 0
    group_id = 0

    for line in lines:
        target = _alignment_key(line)

        if not target:
            continue

        start_pos = record_pos
        group_records = []
        matched = False

        while record_pos < len(records):
            group_records.append(records[record_pos])
            record_pos += 1

            joined = _alignment_key("".join(r["text"] for r in group_records))

            if joined == target:
                matched = True
                break

            # If joined has clearly gone beyond the target, stop.
            # We do not want to consume unrelated OCR nodes.
            if len(joined) > len(target) and not joined.startswith(target):
                break

        if matched:
            for r in group_records:
                r["group_id"] = group_id
                r["group_text"] = line

            groups.append(
                {
                    "group_id": group_id,
                    "group_text": line,
                    "records": group_records,
                }
            )
            group_id += 1

        else:
            # Conservative recovery:
            # If alignment fails, do not create an unsafe multi-node group.
            # Put the next record alone and continue.
            record_pos = start_pos

            if record_pos < len(records):
                r = records[record_pos]
                r["group_id"] = group_id
                r["group_text"] = r["text"]

                groups.append(
                    {
                        "group_id": group_id,
                        "group_text": r["text"],
                        "records": [r],
                    }
                )

                group_id += 1
                record_pos += 1

    # Any leftover records become single-record groups.
    while record_pos < len(records):
        r = records[record_pos]
        r["group_id"] = group_id
        r["group_text"] = r["text"]

        groups.append(
            {
                "group_id": group_id,
                "group_text": r["text"],
                "records": [r],
            }
        )

        group_id += 1
        record_pos += 1

    return groups


# ----------------------------
# Candidate generation
# ----------------------------


# Pipeline order: 34.1
# Description: Builds one SKU candidate from a contiguous window of OCR records.
def make_sku_candidate(window: list[dict], start: int, end: int) -> dict | None:
    raw_text = " ".join(r["text"] for r in window)
    normalized = normalize_sku_candidate(raw_text)

    if normalized is None:
        return None

    bbox = merge_bboxes([r["bbox"] for r in window])

    symbol_only_count = sum(1 for r in window if not any(ch.isdigit() for ch in r["text"]))

    return {
        "start": start,
        "end": end,
        "record_indices": [r["idx"] for r in window],
        "raw_text": raw_text,
        "normalized_text": normalized,
        "digit_len": len(normalized),
        "num_records": len(window),
        "symbol_only_count": symbol_only_count,
        "bbox": bbox,
    }


# Pipeline order: 34
# Description: Finds SKU candidates from OCR records and full-text groups.
def extract_sku_candidates_from_records(
    records: list[dict],
    max_window_size: int = 4,
) -> list[dict]:
    """
    Generate candidates only from contiguous nodes inside one logical group.

    This means:
    - Allowed: ["1004", "334-515"] -> "1004334515"
    - Not allowed: jumping from one newline group to another.
    """

    candidates = []
    n = len(records)

    for start in range(n):
        end_limit = min(n, start + max_window_size)

        for end in range(start + 1, end_limit + 1):
            window = records[start:end]
            cand = make_sku_candidate(window, start, end)

            if cand is not None:
                candidates.append(cand)

    return candidates


# Pipeline order: 34.5
# Description: Selects the best non-overlapping SKU candidates when multiple OCR windows conflict.
def select_non_overlapping_sku_candidates(candidates: list[dict]) -> list[dict]:
    """
    Select best candidates within one group.

    Preference:
    1. 10-digit SKUs over 6-digit SKUs.
    2. Avoid symbol-only OCR nodes.
    3. Prefer fewer OCR nodes if digit length is same.
    4. Prefer smaller bbox if still tied.
    """

    if not candidates:
        return []

    # Pipeline order: 34.6
    # Description: Scores SKU candidates so cleaner and longer candidates win conflicts.
    def score(c):
        return (
            c["digit_len"],
            -c["symbol_only_count"],
            -c["num_records"],
            -(c["bbox"].x2 - c["bbox"].x1),
        )

    candidates = sorted(candidates, key=score, reverse=True)

    selected = []
    used_positions: set[int] = set()

    for cand in candidates:
        positions = set(range(cand["start"], cand["end"]))

        if positions & used_positions:
            continue

        selected.append(cand)
        used_positions.update(positions)

    selected.sort(key=lambda c: c["bbox"].x1)

    return selected


# Pipeline order: 34.4
# Description: Finds contiguous OCR-record segments that were not already consumed by selected candidates.
def _available_segments(records: list[dict], used_record_indices: set[int]):
    """
    Split records into contiguous available segments.

    This prevents creating candidates that cross over already-taken records.
    """

    segment = []

    for r in records:
        if r["idx"] in used_record_indices:
            if segment:
                yield segment
                segment = []
        else:
            segment.append(r)

    if segment:
        yield segment


# Pipeline order: 34.7
# Description: Converts an internal SKU candidate object into the final OCR result dictionary format.
def _candidate_to_result(cand: dict) -> dict:
    return {
        "text": cand["normalized_text"],
        "raw_text": cand["raw_text"],
        "bbox": cand["bbox"],
        "confidence": 1.0,
        "source": "google_ocr_sku_parse",
        # Useful for debugging. Remove if you do not want it in final output.
        "record_indices": cand["record_indices"],
    }


# ----------------------------
# Main parser
# ----------------------------


# Pipeline order: 31
# Description: Main OCR parser that turns Google OCR annotations into validated SKU result dictionaries.
def parse_google_ocr_words(annotations) -> list[dict]:
    """
    Safer Google OCR SKU parser.

    Flow:
    1. Build OCR records from annotations[1:].
    2. Use annotations[0].description split by "\\n" to build logical groups.
    3. First lock clean individual 10-digit SKUs.
    4. Then search only within each logical group for contiguous untaken nodes.
    5. Never merge nodes across newline groups.
    """

    if not annotations:
        return []

    records = build_ocr_records(annotations)

    if not records:
        return []

    groups = build_full_text_groups(annotations, records)

    final_results = []
    used_record_indices: set[int] = set()

    # Step 1:
    # Lock clean individual 10-digit SKUs first.
    # Do not lock 6-digit nodes here because they may be part of a split 10-digit SKU.
    for r in records:
        normalized = normalize_sku_candidate(r["text"])

        if normalized is None:
            continue

        if len(normalized) != 10:
            continue

        cand = make_sku_candidate([r], start=0, end=1)

        if cand is None:
            continue

        final_results.append(_candidate_to_result(cand))
        used_record_indices.add(r["idx"])

    # Step 2:
    # Search remaining untaken nodes only inside their newline/logical group.
    for group in groups:
        group_records = group["records"]

        for segment in _available_segments(group_records, used_record_indices):
            candidates = extract_sku_candidates_from_records(
                segment,
                max_window_size=4,
            )

            selected = select_non_overlapping_sku_candidates(candidates)

            for cand in selected:
                if any(idx in used_record_indices for idx in cand["record_indices"]):
                    continue

                final_results.append(_candidate_to_result(cand))
                used_record_indices.update(cand["record_indices"])

    # Reading order.
    final_results.sort(key=lambda x: (x["bbox"].y1, x["bbox"].x1))

    return final_results


# ----------------------------
# Multi-angle OCR result selection
# ----------------------------


def _source_priority(source: str) -> int:
    """Lower is better — prefer native upright reads over retries."""
    source = str(source or "")
    if source in ("", "google_ocr_sku_parse"):
        return 0
    if "enhanced" in source:
        return 1
    if "rot5" in source or "rotneg5" in source:
        return 2
    if "rot10" in source or "rotneg10" in source:
        return 3
    if "rot180" in source:
        return 4
    return 5


def sku_result_rank(result: dict) -> tuple:
    text = str(result.get("text") or "")
    digit_len = len(text) if text.isdigit() else 0
    return (digit_len, -_source_priority(str(result.get("source") or "")))


def normalize_deskew_angle(angle: float) -> float:
    """Map a text baseline angle to the smallest in-plane rotation that levels it."""
    normalized = float(angle)
    while normalized > 90.0:
        normalized -= 180.0
    while normalized < -90.0:
        normalized += 180.0
    return normalized


def estimate_skew_degrees_from_annotations(annotations: list | None) -> float | None:
    """
    Estimate label skew from Google OCR word/line polygons.

    Uses the top edge of each word box (vertex 0 → vertex 1) and returns the
    median baseline angle in degrees. Positive = clockwise tilt in image coords.
    """
    if not annotations or len(annotations) < 2:
        return None

    angles: list[float] = []
    for item in annotations[1:]:
        vertices = _get_vertices(item)
        if len(vertices) < 2:
            continue
        x0 = _vertex_value(vertices[0], "x")
        y0 = _vertex_value(vertices[0], "y")
        x1 = _vertex_value(vertices[1], "x")
        y1 = _vertex_value(vertices[1], "y")
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        if abs(dx) + abs(dy) < 4.0:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)))

    if not angles:
        return None

    angles.sort()
    return angles[len(angles) // 2]


def deskew_rotation_for_baseline(skew_degrees: float) -> float:
    """Rotation to apply to the image so a tilted baseline reads horizontally."""
    return normalize_deskew_angle(-float(skew_degrees))


def is_strong_sku_read(results: list[dict]) -> bool:
    """True when OCR already found a valid 6- or 10-digit SKU."""
    for result in results:
        text = str(result.get("text") or "")
        if text.isdigit() and len(text) in (6, 10):
            return True
    return False


def merge_multi_pass_sku_results(passes: list[list[dict]]) -> list[dict]:
    """Merge OCR passes, keeping the best source for each unique SKU text."""
    best_by_text: dict[str, dict] = {}

    for pass_results in passes:
        for result in pass_results:
            text = str(result.get("text") or "")
            if not text:
                continue
            existing = best_by_text.get(text)
            if existing is None or sku_result_rank(result) > sku_result_rank(existing):
                best_by_text[text] = result

    merged = list(best_by_text.values())
    merged.sort(
        key=lambda item: (
            item.get("bbox").y1 if item.get("bbox") is not None else 0,
            item.get("bbox").x1 if item.get("bbox") is not None else 0,
        )
    )
    return merged


def primary_sku_suggestion(results: list[dict]) -> str:
    if not results:
        return ""
    best = max(results, key=sku_result_rank)
    return str(best.get("text") or "")

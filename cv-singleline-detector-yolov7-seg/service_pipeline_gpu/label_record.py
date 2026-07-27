"""
label_record.py
---------------
Data structures for the 2-model YOLOv7 pipeline:
  Stage 1 — Label detection    (YOLOv7        → best.pt)
  Stage 2 — Strip segmentation (YOLOv7-seg    → segmentation.pt)

Removed vs. original:
  - rotation_count / crop_image  (no orient stage)
  - seg_fl/sl/q masks + candidates (seg model in this project predicts different classes)
  - first_line, second_line, quantity, reading_* (no OCR/digit stage)
  - digit_detections, boundary_source, label_boundary_used (digit stage artefacts)
  - DigitInferenceResult (no digit model)

Status progression:
  "detected"  → Stage 1 complete  (bbox, score, children set)
  "stitched"  → label crop placed into a strip (strip_index, slot, offsets set)
  "segmented" → Stage 2 complete  (seg masks resolved onto the record)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


# ── Core record ───────────────────────────────────────────────────────────────

@dataclass
class LabelRecord:
    """
    One record per detected label in a shelf image.
    Enriched stage-by-stage as it flows through the pipeline.

    Convention: fields for a given stage are only valid once `status` has
    reached that stage. Downstream code must check `record.status` before
    reading stage-specific fields.
    """

    # ── Identity — always valid ───────────────────────────────────────────────
    label_id:        int          # sequential per image, sorted left→right by x
    source_image:    str          # base image filename or "in-memory"

    # ── Stage 1: Detection ───────────────────────────────────────────────────
    original_bbox:   List[float]              # [x1, y1, x2, y2] in base image coords
    detection_score: float                    # confidence from YOLOv7 NMS output
    has_children:    bool          = False    # True if FL/SL/Q sub-elements were associated
    children:        Optional[List[Dict]] = None  # raw child dicts; dropped after stitching

    # ── Stage 2 prep: Strip assembly ─────────────────────────────────────────
    strip_index:       Optional[int]   = None  # which strip this label lives in
    slot_in_strip:     Optional[int]   = None  # 0-based left-to-right position in strip
    x_offset_in_strip: Optional[int]   = None  # pixel x-start of crop inside strip image
    x_end_in_strip:    Optional[int]   = None  # pixel x-end   of crop inside strip image
    scale_factor:      Optional[float] = None  # resize ratio applied when building the strip

    # Labels that can't be cleanly cropped are excluded before strip assembly
    excluded:          bool          = False
    exclusion_reason:  Optional[str] = None

    # ── Stage 2: Segmentation masks ──────────────────────────────────────────
    # Overall label boundary — enclosing rect + full binary mask of the label region
    seg_label_bbox:        Optional[List[float]] = None  # [x1, y1, x2, y2] enclosing rect
    seg_label_mask:        Optional[np.ndarray]  = None  # binary mask, strip resolution
    seg_label_candidates:  Optional[List[Dict]]  = None  # all candidates before best-pick
    #   ↑ candidates structure: [{"mask": np.ndarray, "bbox": List[float], "confidence": float}]

    # Sub-region masks — add/rename to match your seg model's actual output classes.
    # Example below assumes the seg model predicts a single "text_region" class in
    # addition to the "label" class. If your model has more classes, add more fields.
    seg_region_mask:       Optional[np.ndarray]  = None  # primary text/content area mask
    seg_region_candidates: Optional[List[Dict]]  = None  # candidates before best-pick

    # ── Status & diagnostics ──────────────────────────────────────────────────
    status:       str           = "detected"  # "detected" | "stitched" | "segmented"
    fail_reason:  Optional[str] = None        # set if seg association fails for this label
    diagnostics:  Optional[Dict] = None       # optional per-label debug dict (timing, counts, etc.)

    def __repr__(self) -> str:
        parts = [f"Label#{self.label_id}[{self.status}]"]
        parts.append(f"bbox={[round(c, 1) for c in self.original_bbox]}")
        if self.excluded:
            parts.append("EXCLUDED")
            if self.exclusion_reason:
                parts.append(self.exclusion_reason)
        if self.strip_index is not None:
            parts.append(f"strip={self.strip_index} slot={self.slot_in_strip}")
        if self.seg_label_bbox is not None:
            parts.append(f"seg_bbox={[round(c, 1) for c in self.seg_label_bbox]}")
        if self.seg_label_mask is not None:
            parts.append(f"seg_label_mask=<{self.seg_label_mask.shape}>")
        if self.seg_region_mask is not None:
            parts.append(f"seg_region_mask=<{self.seg_region_mask.shape}>")
        if self.fail_reason:
            parts.append(f"fail={self.fail_reason}")
        return " | ".join(parts)


# ── Strip container ───────────────────────────────────────────────────────────

@dataclass
class StripInfo:
    """
    Holds one stitched strip image and all LabelRecords placed in it.
    One StripInfo per strip row.
    """
    strip_index:   int
    strip_image:   np.ndarray           # H × W × 3, BGR
    label_records: List[LabelRecord]    # ordered left-to-right within the strip
    num_labels:    int = 0

    def __post_init__(self):
        self.num_labels = len(self.label_records)


# ── Detection result types ────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Single bounding-box detection from YOLOv7 (Stage 1 raw output)."""
    bbox:       List[float]                      # [x1, y1, x2, y2]
    class_id:   int
    confidence: float
    centroid:   Tuple[float, float] = (0.0, 0.0) # (cx, cy) — call compute_centroid() after init

    def compute_centroid(self) -> "DetectionResult":
        self.centroid = (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )
        return self


@dataclass
class SegDetectionResult:
    """Single detection from YOLOv7-seg (Stage 2 raw output) — includes binary mask."""
    bbox:       List[float]                      # [x1, y1, x2, y2]
    class_id:   int
    confidence: float
    mask:       np.ndarray                       # binary mask at original strip resolution
    centroid:   Tuple[float, float] = (0.0, 0.0)

    def compute_centroid(self) -> "SegDetectionResult":
        self.centroid = (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )
        return self


# ── Inference result containers ───────────────────────────────────────────────
# GPU phase populates these; CPU association logic reads them.

@dataclass
class SegInferenceResult:
    """Raw output from YOLOv7-seg for one strip image."""
    strip_index: int
    detections:  List[SegDetectionResult] = field(default_factory=list)
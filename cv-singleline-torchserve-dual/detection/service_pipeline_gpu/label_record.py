"""Detection result records used by Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LabelRecord:
    """One detected label from a full shelf image."""

    label_id: int
    source_image: str
    original_bbox: List[float]
    detection_score: float
    has_children: bool = False
    children: Optional[List[Dict]] = None
    excluded: bool = False
    exclusion_reason: Optional[str] = None
    status: str = "detected"

    def __repr__(self) -> str:
        return (
            f"Label#{self.label_id}[{self.status}] "
            f"bbox={[round(value, 1) for value in self.original_bbox]} "
            f"score={self.detection_score:.3f}"
        )

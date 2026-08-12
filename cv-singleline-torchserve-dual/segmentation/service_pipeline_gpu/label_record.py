"""Strip records consumed by the segmentation stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np


@dataclass
class StripInfo:
    """One shelf-strip image sent to the segmentation model."""

    strip_index: int
    strip_image: np.ndarray
    label_records: List[Any] = field(default_factory=list)
    num_labels: int = 0

    def __post_init__(self) -> None:
        self.num_labels = len(self.label_records)

import os
import socket
from typing import Optional, Any
from dataclasses import dataclass, field


gce_name = socket.gethostname()

# Environment variables

subscription_id = os.environ.get("SUBSCRIPTION_ID", "x")
project_id = os.environ.get("GCP_PROJECT", "x")
topic_id = os.environ.get("TOPIC_NAME", "x")
experience = os.environ.get("EXPERIENCE", "x")
sub_experience = os.environ.get("SUB_EXPERIENCE", "x")
application = os.environ.get("APPLICATION", "x")
environment = os.environ.get("ENVIRONMENT", "x")

vertex_project_id = os.environ.get("VERTEX_PROJECT", "x")
vertex_location = os.environ.get("VERTEX_LOCATION", "x")
vertex_yolov7_OD_endpoint_id = os.environ.get("VERTEX_YOLOV7_OD_ENDPOINT_ID", "x")
vertex_yolov7_SEG_endpoint_id = os.environ.get("VERTEX_YOLOV7_SEG_ENDPOINT_ID", "x")

# vertex_endpoint_id = os.environ["VERTEX_ENDPOINT_ID"]
# endpoint_base_name = f"projects/{vertex_project_id}/locations/{vertex_location}/endpoints/{vertex_endpoint_id}"
# psc_base_endpoint_hostname = os.environ["VERTEX_PSC_ENDPOINT_HOSTNAME"]

od_endpoint_base_name = f"projects/{vertex_project_id}/locations/{vertex_location}/endpoints/{vertex_yolov7_OD_endpoint_id}"
seg_endpoint_base_name = f"projects/{vertex_project_id}/locations/{vertex_location}/endpoints/{vertex_yolov7_SEG_endpoint_id}"

od_psc_base_endpoint_hostname = os.environ.get("VERTEX_YOLOV7_OD_PSC_ENDPOINT_HOSTNAME", "x")
seg_psc_base_endpoint_hostname = os.environ.get("VERTEX_YOLOV7_SEG_PSC_ENDPOINT_HOSTNAME", "x")

max_flow_control_limit = 1
# store_list_exp = json.loads(os.getenv("EXPERIMENTAL_STORE_LIST"))

OD_MODEL_NAME = "detection" # "DETECTION"
SEG_MODEL_NAME = "segmentation" # "SEGMENTATION"

OD_CLASS_NAMES = {
    0: "Pallet",
    1: "RDC",
    2: "Printed_on_Box",
    3: "Handwritten",
    4: "Multiline_Label",
    5: "Other",
}

# For record, not used.
SEG_CLASS_NAMES = {
    0: "RDC_SKU",
    1: "Pallet_SKU",
    2: "Printed_on_Box_SKU",
    3: "Handwritten_SKU",
}


OD_CONF_THRESHOLD = float(os.environ.get("OD_CONF_THRESHOLD", "0.25"))
SEG_CONF_THRESHOLD = float(os.environ.get("SEG_CONF_THRESHOLD", "0.45"))
BBOX_BUFFER_PCT = float(os.environ.get("BBOX_BUFFER_PCT", "0.12"))
MAX_STRIPS = int(os.environ.get("MAX_STRIPS", "5"))
IOU_WORD_THRESHOLD = float(os.environ.get("IOU_WORD_THRESHOLD", "0.10"))
SKIP_CLASSES = (4,)  # Multiline_Label - we won't attempt OCR on these since they're not single lines

# STATUS MESSAGES
CROPS_NOT_DETECTED = "CROPS_NOT_DETECTED"
CROPS_ENDPOINT_FAILED = "CROPS_ENDPOINT_FAILED"
SEG_ENDPOINT_FAILED = "SEG_ENDPOINT_FAILED"
IMAGE_NOT_DOWNLOADED = "IMAGE_NOT_DOWNLOADED"
FINISHED = "FINISHED"
NO_SKUS_FOUND = "NO_SKUS_FOUND"


# New inference pipeline dataclasses
# Pipeline order: 00
# Description: Represents a rectangular image region used for detections, OCR words, crops, and coordinate mapping.
@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = ""

    # Pipeline order: 16
    # Description: Expands a detection box by a percentage while keeping it inside image bounds.
    def apply_buffer(self, img_w: int, img_h: int, pct: float = 0.12) -> "BoundingBox":
        bw = self.x2 - self.x1
        bh = self.y2 - self.y1
        pad_x = int(bw * pct)
        pad_y = int(bh * pct)
        return BoundingBox(
            x1=max(0, self.x1 - pad_x),
            y1=max(0, self.y1 - pad_y),
            x2=min(img_w, self.x2 + pad_x),
            y2=min(img_h, self.y2 + pad_y),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )


# @dataclass
# class SegmentationMask:
#     mask: np.ndarray
#     class_id: int
#     class_name: str
#     confidence: float = 1.0


# @dataclass
# class OCRResult:
#     text: str
#     class_name: str
#     confidence: float = 1.0
#     source: str = ""


# Pipeline order: 00
# Description: Represents one parsed OCR SKU result with its detection metadata and original-image bbox.
@dataclass
class OCRWordResult:
    text: str
    raw_text: str
    det_id: int
    class_id: int
    class_name: str
    source: str
    strip_bbox: BoundingBox
    original_bbox: BoundingBox
    confidence: float = 1.0


# Pipeline order: 00
# Description: Holds the full inference result for one image, including OCR results and metadata.
@dataclass
class PipelineResult:
    raw_ocr_results: list[OCRWordResult] = field(default_factory=list)
    fallback_used: bool = False
    strip_fallbacks: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Pipeline order: 00
# Description: Tracks one detected label through crop, strip placement, segmentation, and OCR assignment.
@dataclass
class ClipTrack:
    det_id: int
    class_id: int
    class_name: str
    confidence: float
    orig_bbox: BoundingBox
    buffered_bbox: Optional[BoundingBox] = None

    clip_h: Optional[int] = None
    clip_w: Optional[int] = None
    pad_h: Optional[int] = None
    pad_w: Optional[int] = None
    pad_offset_x: Optional[int] = None
    pad_offset_y: Optional[int] = None

    strip_index: Optional[int] = None
    strip_slot: Optional[int] = None
    strip_bbox: Optional[BoundingBox] = None
    crop_strip_bbox: Optional[BoundingBox] = None

    seg_found: bool = False
    ocr_found: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

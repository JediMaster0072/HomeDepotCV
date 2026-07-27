from typing import Dict, List
import numpy as np
from common_config_gpu import ensure_gpu_only_import_paths
import base64
import io
from typing import Union
import numpy as np
from PIL import Image

ensure_gpu_only_import_paths()

from service_pipeline_gpu.label_record import (  # noqa: E402
    LabelRecord,
    DetectionResult,
    SegDetectionResult,
    SegInferenceResult,
)


def numpy_image_to_base64_png(image: np.ndarray) -> str:
    """
    Encode a NumPy image/mask as lossless PNG and return a base64 string.

    Supports:
    - Grayscale: H x W
    - RGB/RGBA: H x W x 3 / H x W x 4
    - uint8 images/masks
    - bool masks, converted to uint8 0/1

    Notes:
    - This preserves pixel values exactly for uint8 images/masks.
    - It avoids OpenCV BGR/RGB confusion.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")

    image = np.asarray(image)

    if image.dtype == np.bool_:
        image = image.astype(np.uint8)

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 or bool image for PNG encoding, got {image.dtype}. "
            "If you have float data, normalize/convert it explicitly before encoding."
        )

    if image.ndim not in (2, 3):
        raise ValueError(f"Expected image shape HxW or HxWxC, got shape {image.shape}")

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {image.shape}")

    # PIL does not need a channel dimension for grayscale
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    image = np.ascontiguousarray(image)

    pil_image = Image.fromarray(image)

    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

    return base64.b64encode(png_bytes).decode("ascii")


def base64_png_to_numpy_image(image_base64: str) -> np.ndarray:
    """
    Decode a base64 PNG string back into a NumPy array.

    Returns:
    - Grayscale PNG -> H x W uint8
    - RGB PNG -> H x W x 3 uint8
    - RGBA PNG -> H x W x 4 uint8
    """

    if not isinstance(image_base64, str):
        raise TypeError(f"Expected base64 string, got {type(image_base64)}")

    image_bytes = base64.b64decode(image_base64.encode("ascii"))

    with io.BytesIO(image_bytes) as buffer:
        pil_image = Image.open(buffer)
        pil_image.load()  # Force loading before buffer closes

    return np.array(pil_image)


# ── Image helpers ────────────────────────────────────────────────────────────


def encode_image_png(image: np.ndarray) -> str:
    return numpy_image_to_base64_png(image)


def decode_image_png(b64: str) -> np.ndarray:
    return base64_png_to_numpy_image(b64)


def encode_mask_png(mask: np.ndarray) -> str:
    bin_mask = (mask > 0).astype(np.uint8) * 255
    return encode_image_png(bin_mask)


def decode_mask_png(b64: str) -> np.ndarray:
    img = base64_png_to_numpy_image(b64)
    if img.ndim == 3:
        img = img[:, :, 0]  # collapse to grayscale if RGB was returned
    return (img > 0).astype(np.uint8)


# ── LabelRecord serialization ────────────────────────────────────────────────


def label_record_to_dict(record: LabelRecord) -> Dict:
    return {
        "label_id": record.label_id,
        "source_image": record.source_image,
        "original_bbox": [float(x) for x in record.original_bbox],
        "detection_score": float(record.detection_score),
        "has_children": bool(record.has_children),
        "children": record.children,
        "status": record.status,
        "excluded": bool(record.excluded),
        "exclusion_reason": record.exclusion_reason,
    }


def label_record_from_dict(data: Dict) -> LabelRecord:
    return LabelRecord(
        label_id=int(data["label_id"]),
        source_image=str(data["source_image"]),
        original_bbox=[float(x) for x in data["original_bbox"]],
        detection_score=float(data["detection_score"]),
        has_children=bool(data.get("has_children", False)),
        children=data.get("children"),
        status=str(data.get("status", "detected")),
        excluded=bool(data.get("excluded", False)),
        exclusion_reason=data.get("exclusion_reason"),
    )


# ── Segmentation results serialization ──────────────────────────────────────


def seg_results_to_dict(results: List[SegInferenceResult]) -> List[Dict]:
    payload = []
    for item in results:
        payload.append(
            {
                "detections": [
                    {
                        "bbox": [float(x) for x in det.bbox],
                        "class_id": int(det.class_id),
                        "confidence": float(det.confidence),
                        "mask_png": encode_mask_png(det.mask),
                    }
                    for det in item.detections
                ],
            }
        )
    return payload


def seg_results_from_dict(payload: List[Dict]) -> List[SegInferenceResult]:
    out = []
    for item in payload:
        detections = []
        for d in item["detections"]:
            det = SegDetectionResult(
                bbox=[float(x) for x in d["bbox"]],
                class_id=int(d["class_id"]),
                confidence=float(d["confidence"]),
                mask=decode_mask_png(d["mask_png"]),
            )
            det.compute_centroid()
            detections.append(det)
        out.append(SegInferenceResult(strip_index=int(item["strip_index"]), detections=detections))
    return out


# ── Raw detection results serialization (no masks) ──────────────────────────


def detection_results_to_dict(results: List[DetectionResult]) -> List[Dict]:
    return [
        {
            "bbox": [float(x) for x in det.bbox],
            "class_id": int(det.class_id),
            "confidence": float(det.confidence),
        }
        for det in results
    ]


def detection_results_from_dict(payload: List[Dict]) -> List[DetectionResult]:
    out = []
    for d in payload:
        det = DetectionResult(
            bbox=[float(x) for x in d["bbox"]],
            class_id=int(d["class_id"]),
            confidence=float(d["confidence"]),
        )
        det.compute_centroid()
        out.append(det)
    return out

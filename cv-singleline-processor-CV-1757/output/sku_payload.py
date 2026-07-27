import json
from collections import Counter, defaultdict
from datetime import datetime

from common_config import BoundingBox, OCRWordResult


def _datetime_con(datetime_convert: str) -> int:
    rfc_datetime = datetime.strptime(datetime_convert, "%Y-%m-%dT%H:%M:%S.%fZ")
    epoch = datetime.utcfromtimestamp(0)
    return int((rfc_datetime - epoch).total_seconds() * 1000)


class SkuData(object):
    # Pipeline order: 00
    # Description: Holds the final output payload fields for both new and legacy SKU result builders.
    def __init__(self):
        self.store_number = None
        self.aisle_number = None
        self.bay_number = None
        self.source_value = None
        self.ldap_id = None
        self.photo_timestamp = None
        self.photo_location_path = None
        self.process_timestamp = None
        self.process_source = None
        self.filename = None
        self.saved_ts = None
        self.skus_list = []
        self.bounding_boxes = {}
        self.inventory_observations = []
        self.empty_ocr_bounding_boxes = {}


# Pipeline order: 44
# Description: Converts a SkuData object into a JSON-serializable dictionary.
def encoder_sku_data(sku_data):
    if isinstance(sku_data, SkuData):
        return {
            "store_number": sku_data.store_number,
            "aisle_number": sku_data.aisle_number,
            "bay_number": sku_data.bay_number,
            "source_value": sku_data.source_value,
            "ldap_id": sku_data.ldap_id,
            "photo_timestamp": sku_data.photo_timestamp,
            "photo_location_path": sku_data.photo_location_path,
            "process_timestamp": sku_data.process_timestamp,
            "process_source": sku_data.process_source,
            "saved_ts": sku_data.saved_ts,
            "filename": sku_data.filename,
            "skus_list": sku_data.skus_list,
            "bounding_boxes": sku_data.bounding_boxes,
            "inventory_observations": sku_data.inventory_observations,
            "empty_ocr_bounding_boxes": sku_data.empty_ocr_bounding_boxes,
        }
    raise TypeError("Object(sku_data) is not of type SkuData.")


# Pipeline order: 43
# Description: Converts a BoundingBox into the legacy bbox string format expected in the output payload.
def bbox_to_legacy_string(bbox: BoundingBox) -> str:
    return f"{{'{bbox.x1}', '{bbox.y1}', '{bbox.x2}', '{bbox.y2}'}}"


# Pipeline order: 42
# Description: Normalizes SKU text into a zero-padded 10-character output value.
def normalize_sku_text(text: str) -> str:
    return str(text).strip().zfill(10)


def _populate_common_fields(sku_data: SkuData, meta_data, bucket_name: str, file_name: str, process_timestamp: str):
    sku_data.store_number = format(meta_data.get("store_number")).zfill(4)
    sku_data.aisle_number = format(meta_data.get("aisle")).zfill(2)
    sku_data.bay_number = format(meta_data.get("bay")).zfill(3)
    sku_data.source_value = format(meta_data.get("source"))
    sku_data.ldap_id = format(meta_data.get("ldap"))
    sku_data.photo_timestamp = format(meta_data.get("captured_ts"))
    sku_data.photo_location_path = f"gs://{bucket_name}/{file_name}"
    sku_data.process_timestamp = process_timestamp
    sku_data.process_source = "singleline"
    sku_data.filename = file_name
    sku_data.saved_ts = format(meta_data.get("saved_ts"))


# Pipeline order: 41
# Description: Converts new-pipeline OCRWordResult objects into the final SKU JSON payload.
def prepare_sku_result_json_new(
    ocr_results: list[OCRWordResult],
    meta_data,
    bucket_name,
    file_name,
    process_timestamp: str | None = None,
):
    sku_data = SkuData()
    if process_timestamp is None:
        process_timestamp = _datetime_con(str(datetime.now().isoformat()[:-3] + "Z"))

    _populate_common_fields(sku_data, meta_data, bucket_name, file_name, process_timestamp)

    # SKU -> list of bbox strings
    bbox_map = defaultdict(list)
    sku_counter = Counter()

    for item in ocr_results:
        sku_number = normalize_sku_text(item.text)
        sku_counter[sku_number] += 1
        bbox_map[sku_number].append(bbox_to_legacy_string(item.original_bbox))
        sku_data.inventory_observations.append(
            {
                "sku": sku_number,
                "raw_text": str(getattr(item, "raw_text", item.text)),
                "bbox": [
                    item.original_bbox.x1,
                    item.original_bbox.y1,
                    item.original_bbox.x2,
                    item.original_bbox.y2,
                ],
                "confidence": float(getattr(item, "confidence", 1.0)),
                "det_id": getattr(item, "det_id", None),
                "class_name": str(getattr(item, "class_name", "")),
                "source": str(getattr(item, "source", "singleline_ocr")),
            }
        )

    for sku_number, count in sku_counter.items():
        sku_list_data = {
            "sku": sku_number,
            "quantity": str(count),
        }
        sku_data.skus_list.append(sku_list_data)

    for sku_number, boxes in bbox_map.items():
        sku_data.bounding_boxes[sku_number] = ", ".join(boxes)

    json_message = json.dumps(sku_data, default=encoder_sku_data)
    return json_message

import numpy as np
from google.cloud import aiplatform
from services.image_processor import numpy_image_to_base64_png, base64_png_to_numpy_image
from common_config import (
    OD_CLASS_NAMES,
    SEG_CLASS_NAMES,
    OD_MODEL_NAME,
    SEG_MODEL_NAME,
    OD_CONF_THRESHOLD,
    SEG_CONF_THRESHOLD,
    SKIP_CLASSES,
    BoundingBox,
    od_endpoint_base_name,
    seg_endpoint_base_name,
    od_psc_base_endpoint_hostname,
    seg_psc_base_endpoint_hostname,
    gce_name,
)


# Pipeline order: 10.1
# Description: Wraps the private Vertex AI object-detection and segmentation endpoints used by the single-line pipeline.
class VertexModelInferenceService:
    """
    Service class for calling Home Depot Vertex AI models through PSC PrivateEndpoint.

    Supports:
    1. Object detection full-image inference
    2. Segmentation strip inference
    """

    # Pipeline order: 10.1
    # Description: Loads model names, class names, and confidence thresholds from common configuration.
    def __init__(self):

        self.od_model_name = OD_MODEL_NAME
        self.seg_model_name = SEG_MODEL_NAME

        self.od_class_names = OD_CLASS_NAMES
        self.seg_class_names = SEG_CLASS_NAMES

        self.od_conf_threshold = OD_CONF_THRESHOLD
        self.seg_conf_threshold = SEG_CONF_THRESHOLD

    # Pipeline order: 14 and 24.1
    # Description: Encodes an image and calls the configured Vertex AI private endpoint for detection or segmentation.
    def _predict(self, model_name: str, image: np.ndarray, file_name: str, store_number: str):
        predict_result = []

        try:
            if model_name == OD_MODEL_NAME:
                endpoint_yolov7_basename = od_endpoint_base_name
                psc_yolov7_endpoint_hostname = od_psc_base_endpoint_hostname
            else:
                endpoint_yolov7_basename = seg_endpoint_base_name
                psc_yolov7_endpoint_hostname = seg_psc_base_endpoint_hostname

            encoded_image = numpy_image_to_base64_png(image)
            instance = {"file": encoded_image, "model_name": model_name}
            print(f"Vertex Model Online Prediction Started: {model_name} for [{file_name}, {gce_name}, {store_number}]")

            psc_endpoint = aiplatform.PrivateEndpoint(endpoint_name=endpoint_yolov7_basename)
            response = psc_endpoint.predict(instances=[instance], endpoint_override=psc_yolov7_endpoint_hostname)

            if response and response.predictions:
                predict_result = response.predictions

                print(
                    f"Completed Vertex Model Online Prediction: {model_name} for [{file_name}, {gce_name}, {store_number}]\t: {response.predictions}, model version: {response.model_version_id}, model id: {response.deployed_model_id}"
                )
            else:
                print(f"{model_name} Vertex Model Online Prediction No response for [{file_name}, {gce_name}, {store_number}]")
                predict_result = []

        except Exception as exc:
            print(f"[{file_name}] Model Inferencing Failed. Error: {exc}")
            return []

        return predict_result

    # Pipeline order: 12
    # Description: Sends the full image to the object detection endpoint and returns parsed label boxes.
    def predict_detection(
        self,
        image: np.ndarray,
        file_name: str,
        store_number: str,
    ) -> list[BoundingBox]:
        """
        Calls object detection model and returns parsed BoundingBox objects.
        image: np.ndarray  OpenCV image
        file_name: string
        store_number: string
        """
        predict_result = self._predict(model_name=self.od_model_name, image=image, file_name=file_name, store_number=store_number)

        return self.parse_detection_outputs(predict_result)

    # Pipeline order: 23
    # Description: Sends a crop strip to the segmentation endpoint and returns a parsed binary mask.
    def predict_segmentation(
        self,
        image: np.ndarray,
        file_name: str,
        store_number: str,
    ) -> np.ndarray | None:
        """
        Calls Segmentation detection model and returns the binary mask image.
        image: np.ndarray  OpenCV image
        file_name: string
        store_number: string
        Returns: np.ndarray (binary mask) or None on error
        """
        predict_result = self._predict(model_name=self.seg_model_name, image=image, file_name=file_name, store_number=store_number)

        return self.parse_segmentation_outputs(predict_result)

    # Pipeline order: 15
    # Description: Converts raw detection endpoint predictions into filtered BoundingBox objects.
    def parse_detection_outputs(self, predictions: list[dict[str, list | float | int]]) -> list[BoundingBox]:
        """
        Parses object detection output.
        Expected format: [{'detections': [[x1, y1, x2, y2, confidence, class_id], ...]}, ...]
        into list[BoundingBox(x1, y1, x2, y2, class_id, class_name, confidence), ...]
        """
        try:
            if not predictions or not len(predictions):
                print("parse_detection_outputs: Empty predictions list")
                return list()

            if not isinstance(predictions[0], dict):
                print(f"parse_detection_outputs: Expected dict at predictions[0], got {type(predictions[0])}")
                return list()

            detections = predictions[0].get("detections", list())

            if not detections or not len(detections):
                print("parse_detection_outputs: No detections found in predictions[0]")
                return list()

            output: list[BoundingBox] = []

            for idx, det in enumerate(detections):
                try:
                    # Handle list format: [x1, y1, x2, y2, confidence, class_id]
                    if isinstance(det, (list, tuple)) and len(det) >= 6:
                        x1, y1, x2, y2, confidence, class_id = det[0], det[1], det[2], det[3], det[4], det[5]
                    else:
                        print(f"parse_detection_outputs: Detection {idx} has unexpected format: {det}")
                        continue

                    class_id = int(class_id)

                    if class_id in SKIP_CLASSES:
                        print(f"parse_detection_outputs: Skipping class {class_id} (multi-line)")
                        continue

                    confidence = float(confidence)

                    if confidence < self.od_conf_threshold:
                        continue

                    class_name = self.od_class_names.get(class_id, "")

                    box = BoundingBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )

                    output.append(box)
                except (ValueError, IndexError, TypeError) as e:
                    print(f"parse_detection_outputs: Error parsing detection {idx}: {e}")
                    continue

            print(f"parse_detection_outputs: Successfully parsed {len(output)} detections")
            return output

        except Exception as e:
            print(f"parse_detection_outputs: Unexpected error: {e}")
            return list()

    # Pipeline order: 25
    # Description: Extracts and decodes the segmentation mask returned by the segmentation endpoint.
    def parse_segmentation_outputs(self, predictions: list) -> np.ndarray | None:
        """
        Parses segmentation outputs.
        Supported formats:
        1) [{'segmentation': [{'strip_index': 0, 'detections': [...], 'mask': {'img': 'base64_string', 'height': H, 'width': W}}, ...]}, ...]
        2) [{'segmentation': {'seg_results': [{'strip_index': 0, 'detections': [...], 'mask': {'img': 'base64_string', 'height': H, 'width': W}}, ...]}}, ...]

        For both formats, the first segmentation entry is used.
        Returns: binary mask as np.ndarray, or None if parsing fails
        """
        try:
            if not predictions:
                print("parse_segmentation_outputs: Empty or None predictions")
                return None

            if not isinstance(predictions, (list, tuple)):
                print(f"parse_segmentation_outputs: Expected list/tuple, got {type(predictions)}")
                return None

            if len(predictions) == 0:
                print("parse_segmentation_outputs: Empty predictions list")
                return None

            first_pred = predictions[0]
            if not isinstance(first_pred, dict):
                print(f"parse_segmentation_outputs: Expected dict at predictions[0], got {type(first_pred)}")
                return None

            segmentation_node = first_pred.get("segmentation")
            if segmentation_node is None:
                print("parse_segmentation_outputs: No 'segmentation' key in predictions[0]")
                return None

            if isinstance(segmentation_node, dict):
                seg_results = segmentation_node.get("seg_results")
                if not isinstance(seg_results, (list, tuple)):
                    print(f"parse_segmentation_outputs: Expected list/tuple for segmentation['seg_results'], got {type(seg_results)}")
                    return None
                segmentation_items = seg_results
            elif isinstance(segmentation_node, (list, tuple)):
                segmentation_items = segmentation_node
            else:
                print(f"parse_segmentation_outputs: Expected list/tuple or dict for segmentation, got {type(segmentation_node)}")
                return None

            if len(segmentation_items) == 0:
                print("parse_segmentation_outputs: segmentation results are empty")
                return None

            # Use first segmentation element. Current endpoint returns one strip/mask per prediction.
            seg_element = segmentation_items[0]

            if not isinstance(seg_element, dict):
                print(f"parse_segmentation_outputs: Expected dict in segmentation list, got {type(seg_element)}")
                return None

            if "mask" not in seg_element:
                available_keys = list(seg_element.keys()) if isinstance(seg_element, dict) else "N/A"
                print(f"parse_segmentation_outputs: 'mask' key not found in segmentation element. Available keys: {available_keys}")
                return None

            mask_data = seg_element["mask"]
            if not isinstance(mask_data, dict):
                print(f"parse_segmentation_outputs: Expected dict for mask_data, got {type(mask_data)}")
                return None

            if "img" not in mask_data:
                available_keys = list(mask_data.keys())
                print(f"parse_segmentation_outputs: 'img' key not found in mask. Available keys: {available_keys}")
                return None

            base64_img = mask_data["img"]
            if not isinstance(base64_img, str):
                print(f"parse_segmentation_outputs: Expected string for base64 img, got {type(base64_img)}")
                return None

            mask_prediction = base64_png_to_numpy_image(base64_img)

            if mask_prediction is None:
                print("parse_segmentation_outputs: base64_png_to_numpy_image returned None")
                return None

            expected_height = mask_data.get("height")
            expected_width = mask_data.get("width")
            if expected_height is not None and expected_width is not None:
                if mask_prediction.shape[0] != int(expected_height) or mask_prediction.shape[1] != int(expected_width):
                    raise ValueError(
                        "Decoded mask shape does not match metadata "
                        f"(decoded={mask_prediction.shape[:2]}, metadata={(int(expected_height), int(expected_width))})"
                    )

            print(f"parse_segmentation_outputs: Successfully parsed segmentation mask with shape {mask_prediction.shape}")
            return mask_prediction

        except (ValueError, KeyError, TypeError, IndexError) as e:
            print(f"parse_segmentation_outputs: Data format error: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            print(f"parse_segmentation_outputs: Unexpected error: {type(e).__name__}: {e}")
            return None

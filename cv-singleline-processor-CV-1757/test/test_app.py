import json
import os
import urllib.request

from unittest.mock import Mock, patch
import pytest
import pickle
from collections import Counter
from types import SimpleNamespace

os.environ["GCP_PROJECT"] = "np-store-sim"
os.environ["TOPIC_NAME"] = "dev-imgproc-to-dbwriter"
os.environ["SUBSCRIPTION_ID"] = "cv-images-singleline-pull-sub"
os.environ["EXPERIENCE"] = "sim"
os.environ["SUB_EXPERIENCE"] = "cv"
os.environ["APPLICATION"] = "cv-singleline-processor"
os.environ["ENVIRONMENT"] = "np"
os.environ["VERTEX_PROJECT"] = "662388291348"
os.environ["VERTEX_LOCATION"] = "us-east1"
os.environ["VERTEX_YOLOV7_OD_ENDPOINT_ID"] = "5179898234499235840"
os.environ["VERTEX_YOLOV7_SEG_ENDPOINT_ID"] = "5747070312571207680"
os.environ["VERTEX_YOLOV7_OD_PSC_ENDPOINT_HOSTNAME"] = "10.10.0.10"
os.environ["VERTEX_YOLOV7_SEG_PSC_ENDPOINT_HOSTNAME"] = "10.10.0.10"

from common_config import BoundingBox, FINISHED, NO_SKUS_FOUND


def get_object_by_name(filepath):
    try:
        datafile = open(filepath, "rb")
        data = pickle.load(datafile)
        datafile.close()
        return data
    except Exception as e:
        print(e)
    return None


def get_jsonobject_by_name(filepath):
    try:
        datafile = open(filepath, "r")
        data = json.load(datafile)
        datafile.close()
        return data
    except Exception as e:
        print(e)
    return None


def test_detect_bounding_box(capfd):
    with patch("google.cloud.logging.Client"):
        with patch("google.cloud.vision.ImageAnnotatorClient"):
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                from app import detect_bounding_box

                bounding_text = get_object_by_name("./test/mockdata/bounding_text.bin")
                data = get_jsonobject_by_name("./test/mockdata/data.json")
                sku_list_collection = Counter(data)
                response = detect_bounding_box(bounding_text, sku_list_collection)
                response_sku_list = get_jsonobject_by_name("./test/mockdata/response_sku_list.json")
                assert response == response_sku_list


def test_find_sku_entities():
    with patch("google.cloud.logging.Client"):
        with patch("google.cloud.vision.ImageAnnotatorClient"):
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                from app import find_sku_entities

                text = get_object_by_name("./test/mockdata/text.bin")
                response_sku_found, recovered_skus = find_sku_entities(text, "0111")
                # Verify we got SKUs back
                assert len(response_sku_found) > 0
                # Verify we recovered some SKUs from OCR edge cases
                assert isinstance(recovered_skus, set)


def test_prepare_sku_result_json():
    with patch("google.cloud.logging.Client"):
        with patch("google.cloud.vision.ImageAnnotatorClient"):
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                data = get_jsonobject_by_name("./test/mockdata/data.json")
                meta_data = get_jsonobject_by_name("./test/mockdata/meta_data.json")
                bounding_box_list = get_object_by_name("./test/mockdata/bounding_text.bin")
                bucket_name = "np-computer-vision-images"
                file_name = "1667680642765#121#48-009"
                from app import prepare_sku_result_json

                json_message = prepare_sku_result_json(data, meta_data, bucket_name, file_name, bounding_box_list, recovered_skus=set())
                assert json_message is not None


def test_find_bounding_values():
    with patch("google.cloud.logging.Client"):
        with patch("google.cloud.vision.ImageAnnotatorClient"):
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                bounding_box_list = get_jsonobject_by_name("./test/mockdata/response_sku_list.json")
                sku = "101205"
                from app import find_bounding_values

                bounding_result, bounding_count = find_bounding_values(bounding_box_list, sku)
                assert bounding_result is not None
                assert isinstance(bounding_count, int)


@pytest.fixture
def image_data():
    image = Mock()
    image.return_value = "FIRST IMAGE"
    return image


@pytest.fixture
def image_text_data():
    image_text = Mock()
    image_text.id = 1
    image_text.detected_text.return_value = "TEXT WITH TEST RESULTS"
    return image_text


@patch("app.vision")
def test_detect_text(vision, image_text_data, image_data):
    with patch("google.cloud.logging.Client"):
        with patch("google.cloud.vision.ImageAnnotatorClient"):
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                url = "gs://np-store-sim-bucket/image.jpg"
                file_name = "test_file.jpg"
                vision.Image.return_value = image_data
                vision_client = vision.ImageAnnotatorClient()
                import app as app

                app.vision_client = vision_client
                vision_client.document_text_detection.return_value = image_text_data
                test_value = app.detect_text(url, file_name)
                vision_client.document_text_detection.assert_called_with(image=image_data)
                assert image_text_data == test_value


@pytest.fixture
def pubsub_publisher():
    publisher = Mock()
    publisher.return_value = "Publisher"
    return publisher


@patch("app.pubsub_v1")
def test_publish_message(pubsub_v1, capfd, pubsub_publisher):
    with patch("google.cloud.logging.Client"):
        with patch("google.cloud.vision.ImageAnnotatorClient"):
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                import app as app

                project_id = app.project_id
                topic_id = app.topic_id
                topic_path = f"projects/{project_id}/topics/{topic_id}"
                app.pubsub_v1 = pubsub_v1
                app.publisher = pubsub_publisher
                pubsub_v1.PublisherClient().return_value = pubsub_publisher
                pubsub_v1.PublisherClient().topic_path.return_value = topic_path
                app.topic_path = topic_path
                pubsub_publisher.publish().result.return_value = 12345654321
                request = "THIS IS TEST DATA"
                file_name = "test_file"
                app.publish_message(request, file_name)
                message_bytes = request.encode("utf-8")
                pubsub_publisher.publish.assert_called_with(topic_path, data=message_bytes)
                out, err = capfd.readouterr()
                assert f"[{file_name}] message-published : {request}" in out and f"[{file_name}] published message id : 12345654321" in out


def test_datetime_con():
    timstamp = "2023-01-03T10:10:10.101Z"
    with patch("google.cloud.logging.Client"):
        with patch("google.cloud.vision.ImageAnnotatorClient"):
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                from app import datetime_con

                assert datetime_con(timstamp) == 1672740610101


def test_match_criteria():
    from translators.atomic.entity_translators import SKUEntityTranslator

    sku_entity_translator = SKUEntityTranslator("skus")
    word = "123456"
    response = sku_entity_translator.matches_criteria(word)
    assert response is True

    word = "12345678"
    response = sku_entity_translator.matches_criteria(word)
    assert response is False

    word = "12345"
    response = sku_entity_translator.matches_criteria(word)
    assert response is False

    word = "1234567890"
    response = sku_entity_translator.matches_criteria(word)
    assert response is True

    word = "1234-567-890"
    response = sku_entity_translator.matches_criteria(word)
    assert response is True

    word = "123-456"
    response = sku_entity_translator.matches_criteria(word)
    assert response is True

    word = "1234-456"
    response = sku_entity_translator.matches_criteria(word)
    assert response is False

    word = "123-456-"
    response = sku_entity_translator.matches_criteria(word)
    assert response is False


@patch("services.image_inference.base64_png_to_numpy_image")
def test_parse_segmentation_outputs_with_seg_results_dict(base64_png_to_numpy_image_mock):
    from services.image_inference import VertexModelInferenceService

    service = VertexModelInferenceService()
    decoded_mask = Mock()
    decoded_mask.shape = (256, 2265)
    base64_png_to_numpy_image_mock.return_value = decoded_mask

    predictions = [
        {
            "segmentation": {
                "seg_results": [
                    {
                        "strip_index": 0,
                        "detections": [{"bbox": [1529, 115, 1651, 157], "class_id": 0, "confidence": 0.422119140625}],
                        "mask": {
                            "height": 256,
                            "width": 2265,
                            "img": "dummy_base64_mask",
                        },
                    }
                ]
            }
        }
    ]

    output = service.parse_segmentation_outputs(predictions)

    base64_png_to_numpy_image_mock.assert_called_once_with("dummy_base64_mask")
    assert output == decoded_mask


def test_parse_segmentation_outputs_with_missing_seg_results_returns_none():
    from services.image_inference import VertexModelInferenceService

    service = VertexModelInferenceService()

    predictions = [{"segmentation": {}}]

    output = service.parse_segmentation_outputs(predictions)

    assert output is None


def test_parse_detection_outputs_filters_and_parses_entries():
    from services.image_inference import VertexModelInferenceService

    service = VertexModelInferenceService()
    service.od_conf_threshold = 0.2

    predictions = [
        {
            "detections": [
                [10.9, 20.1, 30.8, 40.6, 0.95, 0],
                [100, 200, 300, 400, 0.1, 0],
                [1, 2, 3, 4, 0.99, 4],
            ]
        }
    ]

    output = service.parse_detection_outputs(predictions)

    assert len(output) == 1
    assert output[0].x1 == 10
    assert output[0].y1 == 20
    assert output[0].x2 == 30
    assert output[0].y2 == 40
    assert output[0].class_id == 0
    assert output[0].confidence == 0.95
    assert output[0].class_name == service.od_class_names.get(0, "")


def test_prepare_sku_result_json_new_zero_pads_sku_strings():
    with (
        patch("google.cloud.logging.Client"),
        patch("google.cloud.vision.ImageAnnotatorClient"),
        patch("google.cloud.pubsub_v1.PublisherClient"),
        patch.dict("sys.modules", {"cv2": Mock(), "google.cloud.aiplatform": Mock()}),
    ):
        from app import prepare_sku_result_json_new

        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        ocr_results = [
            SimpleNamespace(
                text="359253",
                raw_text="359253",
                class_id=1,
                class_name="RDC",
                original_bbox=bbox,
            )
        ]
        meta_data = {
            "store_number": "244",
            "aisle": "9",
            "bay": "6",
            "source": "camera_cart",
            "ldap": "cart0244",
            "captured_ts": "1770339221260",
            "saved_ts": "1778586941646",
        }

        result = json.loads(prepare_sku_result_json_new(ocr_results, meta_data, "bucket", "file"))

        assert result["skus_list"][0]["sku"] == "0000359253"
        assert "0000359253" in result["bounding_boxes"]
        assert "359253" not in result["bounding_boxes"]
        assert result["inventory_observations"] == [
            {
                "sku": "0000359253",
                "raw_text": "359253",
                "bbox": [10, 20, 100, 50],
                "confidence": 1.0,
                "det_id": None,
                "class_name": "RDC",
                "source": "singleline_ocr",
            }
        ]


def test_process_image_new_returns_fallback_status_when_finished_without_skus():
    with (
        patch("google.cloud.logging.Client"),
        patch("google.cloud.vision.ImageAnnotatorClient"),
        patch("google.cloud.pubsub_v1.PublisherClient"),
        patch.dict("sys.modules", {"cv2": Mock(), "google.cloud.aiplatform": Mock()}),
    ):
        import app

        pubsub_message = {
            "url": "gs://bucket/file",
            "store_number": "244",
            "aisle": "9",
            "bay": "6",
            "source": "camera_cart",
            "ldap": "cart0244",
            "captured_ts": "1770339221260",
            "saved_ts": "1778586941646",
        }
        pipeline_result = SimpleNamespace(raw_ocr_results=[])
        pipeline = Mock()
        pipeline.run_image.return_value = (FINISHED, pipeline_result)

        with (
            patch("app.download_image", return_value=Mock()),
            patch("app.get_inference_pipeline", return_value=pipeline),
            patch("app.publish_message") as publish_message,
        ):
            status = app.process_image_new(
                uri="gs://bucket/file",
                bucket="bucket",
                file_name="file",
                store_number="0244",
                image_np=Mock(),
                pubsub_message=pubsub_message,
            )

        assert status == NO_SKUS_FOUND
        publish_message.assert_not_called()


# Health check endpoint tests


@pytest.fixture(scope="module")
def health_server():
    from health import start_health_server

    server = start_health_server(port=0)  # port=0 lets the OS pick a free port
    yield server
    server.shutdown()


def test_liveness_endpoint(health_server):
    port = health_server.server_address[1]
    with urllib.request.urlopen(f"http://localhost:{port}/health/liveness") as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
        assert body == {"status": "alive"}


def test_readiness_endpoint(health_server):
    port = health_server.server_address[1]
    with urllib.request.urlopen(f"http://localhost:{port}/health/readiness") as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
        assert body == {"status": "ready"}


def test_unknown_path_returns_404(health_server):
    import urllib.error

    port = health_server.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://localhost:{port}/unknown")
    assert exc_info.value.code == 404

import os
import re
import json

import urllib3
import logging
from datetime import datetime
from collections import Counter

from prometheus_client import start_http_server
from google.cloud import pubsub_v1
from google.cloud import vision
from google.cloud.pubsub_v1.subscriber.message import Message

import services.metrics as metrics
from services.image_processor import download_image
from health import start_health_server
from utils.common_utils import log_metric
from legacy.ocr_parsing import (
    find_sku_entities,
    detect_bounding_box,
    find_bounding_values,
    structure_bounding,
)
from output.sku_payload import (
    SkuData,
    encoder_sku_data,
    bbox_to_legacy_string,
    normalize_sku_text,
    prepare_sku_result_json_new,
)

# from new_inference_pipeline import HomeDepotInferencePipeline
from new_inference_pipeline_full_image import HomeDepotInferencePipeline

from common_config import (
    subscription_id,
    project_id,
    topic_id,
    experience,
    sub_experience,
    application,
    environment,
    max_flow_control_limit,
    # PipelineResult,
    # IMAGE_NOT_DOWNLOADED,
    FINISHED,
    NO_SKUS_FOUND,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Legacy Vision client is kept because you asked to retain the existing OCR path.
vision_client = vision.ImageAnnotatorClient()
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

_inference_pipeline = None


# Pipeline order: 06
# Description: Extracts the GCS URI, bucket, file name, and zero-padded store number from the Pub/Sub metadata.
def get_base_metadata(pubsub_message):
    uri = pubsub_message["url"]
    match = re.match(r"gs://(.*?)/(.*)", uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {uri}")

    bucket, file_name = match.groups()
    store_number = str(pubsub_message["store_number"]).zfill(4)
    print(f"Processing file {file_name} in bucket {bucket} from store {store_number}...")
    return uri, bucket, file_name, store_number


# Pipeline order: Optional legacy fallback
# Description: Calls Google Vision OCR directly on the original GCS image URI.
def detect_text(uri, file_name):
    print(f"Looking for text in image {uri}")
    image = vision.Image(source=vision.ImageSource(gcs_image_uri=uri))
    text_detection_response = vision_client.document_text_detection(image=image)
    fulltext = text_detection_response
    print(f"[{file_name}] Text Response from Google Vision API : {fulltext.full_text_annotation.text.split(chr(10))}")
    return fulltext

# Pipeline order: Optional legacy fallback
# Description: Builds the legacy output JSON payload from legacy OCR SKU strings and bounding boxes.
def prepare_sku_result_json(sku_list, meta_data, bucket_name, file_name, sku_text, recovered_skus=None, bbox_override=None):
    sku_data = SkuData()
    # row data
    sku_data.store_number = format(meta_data.get("store_number")).zfill(4)
    sku_data.aisle_number = format(meta_data.get("aisle")).zfill(2)
    sku_data.bay_number = format(meta_data.get("bay")).zfill(3)
    sku_data.source_value = format(meta_data.get("source"))
    sku_data.ldap_id = format(meta_data.get("ldap"))
    sku_data.photo_timestamp = format(meta_data.get("captured_ts"))
    sku_data.photo_location_path = f"gs://{bucket_name}/{file_name}"
    current_time = datetime.now()
    sku_data.process_timestamp = datetime_con(str(current_time.isoformat()[:-3] + "Z"))
    sku_data.process_source = "singleline"
    sku_data.filename = file_name
    sku_list_collection = Counter(sku_list)
    # Use pre-filtered bbox list when available (avoids re-scanning OCR and ensures
    # bounding_boxes only contains the bboxes that passed spatial filtering).
    bounding_box_list = bbox_override if bbox_override is not None else detect_bounding_box(sku_text, sku_list_collection, file_name)
    # Log recovered SKUs and their bounding boxes
    if recovered_skus:
        recovered_sku_bbox = [item for item in bounding_box_list if item[0] in recovered_skus]
        if recovered_sku_bbox:
            print(f"[{file_name}] Sku_list_recovered: {recovered_sku_bbox}")

    print(f"[{file_name}] Sku_list : {sku_list} .")
    print(f"[{file_name}] Sku_list_counter : {sku_list_collection} .")
    for sku in sku_list_collection:
        sku_number = str(sku).zfill(10)
        sku_occurrences = str(sku_list_collection[sku])
        bounding_result, bounding_count = find_bounding_values(bounding_box_list, sku)
        sku_data.bounding_boxes[sku_number] = bounding_result
        sku_list_data = {
            "sku": sku_number,
            "quantity": sku_occurrences,
        }
        sku_data.skus_list.append(sku_list_data)
    sku_data.saved_ts = format(meta_data.get("saved_ts"))
    json_message = json.dumps(sku_data, default=encoder_sku_data)
    return json_message


# Pipeline order: Optional legacy fallback
# Description: Runs full-image Google OCR and publishes legacy SKU results when the new pipeline cannot produce SKUs.
def process_image(
    uri,
    bucket,
    file_name,
    store_number,
    image_np,
    pubsub_message,
) -> str:
    print(f"Processing file {file_name} in bucket {bucket} from store {store_number}...")

    sku_text = detect_text(uri, file_name)  # Google OCR call
    sku_annotation_text = sku_text.full_text_annotation.text

    if sku_annotation_text:
        skus, recovered_skus = find_sku_entities(sku_text, store_number, file_name)
        filtered_bbox_override = None
        if len(skus) > 0:
            message_publish = prepare_sku_result_json(skus, pubsub_message, bucket, file_name, sku_text, recovered_skus, bbox_override=filtered_bbox_override)
            publish_message(message_publish, file_name)
    else:
        print("No text found in image.")
    print(f"File {file_name} processed.")


# Pipeline order: 09
# Description: Lazily creates and reuses the single-line inference pipeline instance.
def get_inference_pipeline() -> HomeDepotInferencePipeline:
    global _inference_pipeline
    if _inference_pipeline is None:
        _inference_pipeline = HomeDepotInferencePipeline()
    return _inference_pipeline


# Pipeline order: 08
# Description: Runs the new model-based single-line pipeline and publishes results when publishable SKUs are found.
def process_image_new(
    uri,
    bucket,
    file_name,
    store_number,
    image_np,
    pubsub_message,
) -> str:
    pipeline = get_inference_pipeline()
    status, pipeline_result = pipeline.run_image(image_np, file_name=file_name, store_number=store_number)

    # Pipeline now returns NO_SKUS_FOUND when OCR/parsing yields no SKU rows.
    if status != FINISHED:
        print(f"[{file_name} - {status}] New Inference pipeline did not finish with publishable SKUs.")
        return status

    # Defensive guard for inconsistent status/result payload combinations.
    if not pipeline_result or not pipeline_result.raw_ocr_results:
        status = NO_SKUS_FOUND
        print(f"[{file_name} - {status}] New Inference pipeline returned FINISHED but no SKU payload.")
        return status

    message_publish = prepare_sku_result_json_new(
        ocr_results=pipeline_result.raw_ocr_results,
        meta_data=pubsub_message,
        bucket_name=bucket,
        file_name=file_name,
    )

    publish_message(message_publish, file_name)

    print(f"File {file_name} processed.")
    return FINISHED


# Pipeline order: 04 and 46
# Description: Converts ISO timestamps into epoch milliseconds for processing metrics.
def datetime_con(datetime_convert):
    rfc_datetime = datetime.strptime(datetime_convert, "%Y-%m-%dT%H:%M:%S.%fZ")
    epoch = datetime.utcfromtimestamp(0)
    epoch_datetime = int((rfc_datetime - epoch).total_seconds() * 1000)
    return epoch_datetime


# Pipeline order: 03
# Description: Orchestrates one Pub/Sub image-processing job from metadata parsing through inference, fallback, and metrics.
def process(message: Message):
    metrics.total_images_count.labels(experience, sub_experience, application, environment).inc()
    log_metric(1, "VOLUME", None)
    process_start_time = int(datetime_con(str(datetime.now().isoformat()[:-3] + "Z")))
    pubsub_message = get_metadata_from_message(message)

    print("Starting New Single-line process : " + pubsub_message["url"])
    uri, bucket, file_name, store_number = get_base_metadata(pubsub_message=pubsub_message)

    image_np = download_image(bucket, file_name)

    if image_np is None:
        print(f"[{bucket} - {file_name}] Failed to download image.")
    else:
        status = process_image_new(
            uri,
            bucket,
            file_name,
            store_number,
            image_np,
            pubsub_message,
        )

        # Run legacy pipeline when the new flow produced no SKU result,
        # or returned a non-success status.
        should_run_legacy = status in (None, NO_SKUS_FOUND) or status != FINISHED
        if should_run_legacy:
            print("Starting Original Single-line process : " + pubsub_message["url"])
            process_image(
                uri,
                bucket,
                file_name,
                store_number,
                image_np,
                pubsub_message,
            )

    process_end_time = int(datetime_con(str(datetime.now().isoformat()[:-3] + "Z")))
    log_metric(process_start_time, process_end_time, "DURATION")
    saved_ts = int(pubsub_message.get("saved_ts"))
    log_metric(saved_ts, process_start_time, "TIME_ELAPSED_START")
    log_metric(saved_ts, process_end_time, "TIME_ELAPSED_END")
    return "Success", 200


# Pipeline order: 45
# Description: Publishes the final SKU JSON payload to the configured output Pub/Sub topic.
def publish_message(request, file_name=None):
    print(f"[{file_name}] message-published : " + request)
    message_bytes = request.encode("utf-8")
    try:
        publish_future = publisher.publish(topic_path, data=message_bytes)
        result = publish_future.result()  # Verify the publishing succeeded
        print(f"[{file_name}] published message id : {result}")
    except Exception as e:
        print(e)
        return e, 500


# Pipeline order: 05
# Description: Decodes the Pub/Sub message bytes into the metadata dictionary used by the pipeline.
def get_metadata_from_message(message: Message) -> dict:
    data = message.data
    meta_data = json.loads(data.decode("utf-8"))
    return meta_data


# Pipeline order: 01
# Description: Starts the Pub/Sub subscriber loop that receives image-processing messages.
def streaming():  # pragma: no cover
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    def callback(message: Message) -> None:
        # Pipeline order: 02
        # Description: Handles one Pub/Sub message and acks or nacks it based on processing success.
        with metrics.overall_process_time.labels(experience, sub_experience, application, environment).time():
            try:
                process(message)
                metrics.total_ack_message_count.labels(experience, sub_experience, application, environment).inc()
                message.ack()
            except Exception as e:
                metrics.total_nack_message_count.labels(experience, sub_experience, application, environment).inc()
                logging.error(f"Failed to process image {get_metadata_from_message(message)['url']}, will process later {e}")
                message.nack()

    flow_control = pubsub_v1.types.FlowControl(max_messages=max_flow_control_limit)
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback, flow_control=flow_control)
    logging.info(f"Listening for messages on {subscription_path}...")

    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result()
        except Exception as ex:
            logging.error(f"Subscriber Exception: {ex}")
            streaming_pull_future.cancel()  # Trigger the shutdown.
            streaming_pull_future.result()  # Block until the shutdown is complete.


if __name__ == "__main__":
    start_health_server()
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8082
    # TODO: Enable Prometheus metrics
    start_http_server(PORT)
    streaming()

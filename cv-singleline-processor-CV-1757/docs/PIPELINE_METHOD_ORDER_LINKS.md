# Pipeline Method Order Links

This file lists the main production single-line pipeline methods in the order they run. The active production path uses `new_inference_pipeline_full_image.py`.

## Main Production Path

1. `app.py::streaming()` starts the Pub/Sub subscriber loop.
2. `app.py::callback(message)` handles one Pub/Sub message and acks or nacks it.
3. `app.py::process(message)` orchestrates one image-processing job.
4. `app.py::datetime_con(...)` converts timestamps for metrics.
5. `app.py::get_metadata_from_message(...)` decodes the Pub/Sub message JSON.
6. `app.py::get_base_metadata(...)` extracts the GCS URI, bucket, file name, and store number.
7. `services/image_processor.py::download_image(...)` downloads the image from GCS.
8. `services/image_processor.py::storage_client()` creates or reuses the GCS client.
9. `services/image_processor.py::convert_pil_to_numpy(...)` converts the downloaded image into a NumPy array.
10. `app.py::process_image_new(...)` runs the new model-based single-line pipeline.
11. `app.py::get_inference_pipeline()` creates or reuses the pipeline instance.
12. `new_inference_pipeline_full_image.py::HomeDepotInferencePipeline.__init__(...)` initializes pipeline settings, Vertex access, and OCR.
13. `new_inference_pipeline_full_image.py::HomeDepotInferencePipeline.run_image(...)` runs the full active inference pipeline for one image.

## Detection Stage

14. `services/image_inference.py::predict_detection(...)` sends the full image to the object detection endpoint.
15. `services/image_inference.py::_predict(...)` calls the Vertex AI private endpoint.
16. `services/image_processor.py::numpy_image_to_base64_png(...)` encodes the image for the endpoint request.
17. `services/image_inference.py::parse_detection_outputs(...)` converts raw detections into `BoundingBox` objects.
18. `common_config.py::BoundingBox.apply_buffer(...)` expands each detected bbox before cropping.
19. `new_inference_pipeline_full_image.py::crop(...)` extracts the buffered crop from the original image.
20. `pipeline/detection_stage.py::preprocess(...)` cleans up each crop before segmentation.
21. `pipeline/detection_stage.py::deblur(...)` sharpens the crop.
22. `pipeline/detection_stage.py::denoise(...)` denoises the crop.

## Strip And Segmentation Stage

23. `pipeline/strips.py::group_clip_items(...)` groups crops into strip batches.
24. `pipeline/strips.py::create_strip(...)` builds one horizontal strip from grouped crops.
25. `pipeline/strips.py::center_pad(...)` pads each crop into a common strip slot size.
26. `pipeline/segmentation_stage.py::mask_single_strip(...)` runs segmentation for one strip and stores masked crop data.
27. `pipeline/segmentation_stage.py::predict_binary_strip_mask(...)` calls segmentation and normalizes the returned mask.
28. `services/image_inference.py::predict_segmentation(...)` sends the strip image to the segmentation endpoint.
29. `services/image_inference.py::_predict(...)` calls the Vertex AI private endpoint for segmentation.
30. `services/image_inference.py::parse_segmentation_outputs(...)` extracts the segmentation mask from the endpoint response.
31. `services/image_processor.py::base64_png_to_numpy_image(...)` decodes a base64 PNG mask into a NumPy array.
32. `pipeline/segmentation_stage.py::process_binary_mask_with_rotation(...)` expands rotated mask regions.
33. `pipeline/segmentation_stage.py::apply_mask_to_strip_preserve_unsegmented_clips(...)` masks valid segmented crops while preserving unsegmented crops.
34. `pipeline/segmentation_stage.py::clip_has_segmentation(...)` checks whether a crop has enough mask pixels to trust segmentation.

## Masked Original And OCR Stage

35. `new_inference_pipeline_full_image.py::_build_masked_original_image(...)` reconstructs a full-resolution image with only SKU regions visible.
36. `pipeline/ocr_stage.py::run_google_ocr_words(...)` runs OCR and returns parsed SKU candidates.
37. `pipeline/ocr_stage.py::call_google_ocr_np(...)` calls Google Vision document text detection.
38. `services/image_processor.py::numpy_image_to_base64_png(...)` encodes the masked image bytes for OCR.
39. `utils/google_ocr_utils.py::parse_google_ocr_words(...)` turns Google OCR annotations into validated SKU result dictionaries.
40. `utils/google_ocr_utils.py::build_ocr_records(...)` builds text and bbox records from OCR annotations.
41. `utils/google_ocr_utils.py::bbox_from_google_annotation(...)` converts OCR polygons into `BoundingBox` objects.
42. `utils/google_ocr_utils.py::build_full_text_groups(...)` aligns full-text OCR lines with individual OCR nodes.
43. `utils/google_ocr_utils.py::extract_sku_candidates_from_records(...)` finds candidate SKU windows.
44. `utils/google_ocr_utils.py::normalize_sku_candidate(...)` validates and normalizes SKU text.
45. `utils/google_ocr_utils.py::select_non_overlapping_sku_candidates(...)` chooses the best non-overlapping SKU candidates.
46. `utils/google_ocr_utils.py::_candidate_to_result(...)` converts internal candidates into final OCR result dictionaries.
47. `pipeline/assignment.py::assign_word_to_track_in_original(...)` assigns OCR words back to detected label tracks.

## Output Publishing Stage

48. `output/sku_payload.py::prepare_sku_result_json_new(...)` converts OCR results into the final output JSON.
49. `output/sku_payload.py::normalize_sku_text(...)` zero-pads SKU text for output.
50. `output/sku_payload.py::bbox_to_legacy_string(...)` formats original-image bboxes for the output schema.
51. `output/sku_payload.py::encoder_sku_data(...)` serializes `SkuData` into JSON.
52. `app.py::publish_message(...)` publishes the final SKU JSON to the output Pub/Sub topic.
53. `app.py::datetime_con(...)` converts end timestamps for metrics.
54. `utils/common_utils.py::log_metric(...)` prints service metric lines.

## Legacy Fallback Path

If the new pipeline returns no publishable SKUs or fails with a non-`FINISHED` status, `process(message)` calls the fallback path:

1. `app.py::process_image(...)` runs the legacy full-image OCR flow.
2. `app.py::detect_text(...)` calls Google Vision OCR directly on the original GCS image.
3. `legacy/ocr_parsing.py::find_sku_entities(...)` extracts legacy SKU strings from OCR text.
4. `translators/atomic/entity_translators.py::SKUEntityTranslator.matches_criteria(...)` checks OCR text against SKU regex rules.
5. `translators/atomic/entity_translators.py::SKUEntityTranslator.extract_candidate_skus(...)` recovers plausible SKUs from noisy OCR text.
6. `app.py::prepare_sku_result_json(...)` builds the legacy output JSON.
7. `legacy/ocr_parsing.py::detect_bounding_box(...)` finds OCR bounding boxes for legacy SKU strings.
8. `legacy/ocr_parsing.py::find_bounding_values(...)` gathers bbox values for one SKU.
9. `legacy/ocr_parsing.py::structure_bounding(...)` converts OCR polygon points into legacy bbox order.
10. `app.py::publish_message(...)` publishes the fallback output JSON.

## Active Data Objects

- `common_config.py::BoundingBox` represents detection, crop, and OCR regions.
- `common_config.py::ClipTrack` tracks one detected label through crop, strip, segmentation, and OCR assignment.
- `common_config.py::OCRWordResult` stores one parsed OCR SKU result.
- `common_config.py::PipelineResult` stores all results and metadata for one image.
- `output/sku_payload.py::SkuData` stores the final output payload fields.

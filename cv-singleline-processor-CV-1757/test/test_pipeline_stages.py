import numpy as np

from common_config import BoundingBox, ClipTrack
from pipeline.detection_tracks import build_clip_items, build_detection_tracks
from pipeline.masked_original_builder import build_masked_original_image
from pipeline.ocr_stage import enhance_image_for_ocr, run_google_ocr_words


def test_build_detection_tracks_buffers_crops_and_records_metadata():
    raw_image = np.zeros((20, 20, 3), dtype=np.uint8)
    detection = BoundingBox(
        x1=5,
        y1=5,
        x2=10,
        y2=10,
        confidence=0.9,
        class_id=1,
        class_name="RDC",
    )

    crops, tracks = build_detection_tracks(
        raw_image=raw_image,
        detections=[detection],
        bbox_buffer_pct=0.2,
        crop_fn=lambda image, box: image[box.y1:box.y2, box.x1:box.x2],
        preprocess_fn=lambda crop: crop,
    )

    assert len(crops) == 1
    assert len(tracks) == 1
    assert crops[0].shape == (7, 7, 3)

    track = tracks[0]
    assert track.det_id == 0
    assert track.class_name == "RDC"
    assert track.orig_bbox == detection
    assert track.buffered_bbox == BoundingBox(4, 4, 11, 11, confidence=0.9, class_id=1, class_name="RDC")
    assert track.clip_h == 7
    assert track.clip_w == 7


def test_build_clip_items_pairs_tracks_with_crops():
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    track = ClipTrack(
        det_id=0,
        class_id=1,
        class_name="RDC",
        confidence=0.9,
        orig_bbox=BoundingBox(0, 0, 2, 2),
    )

    items = build_clip_items([crop], [track])

    assert len(items) == 1
    assert items[0]["track"] is track
    assert items[0]["crop"] is crop


def test_build_masked_original_image_reveals_segmented_pixels_only():
    raw_image = np.zeros((10, 10, 3), dtype=np.uint8)
    raw_image[2:6, 2:6] = 40

    track = ClipTrack(
        det_id=0,
        class_id=1,
        class_name="RDC",
        confidence=0.9,
        orig_bbox=BoundingBox(2, 2, 6, 6),
        buffered_bbox=BoundingBox(2, 2, 6, 6),
        crop_strip_bbox=BoundingBox(0, 0, 2, 2),
        seg_found=True,
    )
    masked_clip = np.zeros((2, 2, 3), dtype=np.uint8)
    masked_clip[0, 0] = 255

    canvas = build_masked_original_image(
        raw_image=raw_image,
        all_groups=[[{"track": track, "masked_strip_clip": masked_clip}]],
        background_mode="white",
        mask_roi_pad_px=0,
        small_label_area_threshold=0,
        blur_sigma=51,
    )

    assert np.array_equal(canvas[2, 2], raw_image[2, 2])
    assert np.array_equal(canvas[5, 5], np.array([255, 255, 255], dtype=np.uint8))


def test_enhance_image_for_ocr_keeps_same_shape_and_dtype():
    image = np.full((20, 30, 3), 180, dtype=np.uint8)
    image[8:12, 10:20] = 60

    enhanced = enhance_image_for_ocr(image)

    assert enhanced.shape == image.shape
    assert enhanced.dtype == image.dtype


def test_run_google_ocr_words_retries_with_enhanced_image(monkeypatch):
    calls = []

    def fake_call_google_ocr_np(_client, image):
        calls.append(image)
        return [f"annotations-{len(calls)}"]

    def fake_parse_google_ocr_words(annotations):
        if annotations == ["annotations-1"]:
            return []
        return [
            {
                "text": "1004334515",
                "raw_text": "1004 334-515",
                "bbox": BoundingBox(1, 2, 10, 12),
                "confidence": 1.0,
                "source": "google_ocr_sku_parse",
            }
        ]

    monkeypatch.setattr("pipeline.ocr_stage.call_google_ocr_np", fake_call_google_ocr_np)
    monkeypatch.setattr("pipeline.ocr_stage.parse_google_ocr_words", fake_parse_google_ocr_words)

    results = run_google_ocr_words(object(), np.full((20, 30, 3), 180, dtype=np.uint8))

    assert len(calls) == 2
    assert results[0]["text"] == "1004334515"
    assert results[0]["source"] == "google_ocr_sku_parse_enhanced_retry"


def test_run_google_ocr_words_retries_rot180_and_maps_bbox(monkeypatch):
    calls = []

    def fake_call_google_ocr_np(_client, image):
        calls.append(image)
        return [f"annotations-{len(calls)}"]

    def fake_parse_google_ocr_words(annotations):
        if annotations in (["annotations-1"], ["annotations-2"]):
            return []
        return [
            {
                "text": "1004334515",
                "raw_text": "1004334515",
                "bbox": BoundingBox(2, 3, 12, 8),
                "confidence": 1.0,
                "source": "google_ocr_sku_parse",
            }
        ]

    monkeypatch.setattr("pipeline.ocr_stage.call_google_ocr_np", fake_call_google_ocr_np)
    monkeypatch.setattr("pipeline.ocr_stage.parse_google_ocr_words", fake_parse_google_ocr_words)

    results = run_google_ocr_words(object(), np.full((20, 30, 3), 180, dtype=np.uint8))

    assert len(calls) == 3
    assert results[0]["source"] == "google_ocr_sku_parse_rot180_retry"
    assert results[0]["bbox"] == BoundingBox(18, 12, 28, 17)

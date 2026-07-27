from utils.crop_preprocess import best_display_angle_from_rotation_hints
from utils.google_ocr_utils import (
    deskew_rotation_for_baseline,
    estimate_skew_degrees_from_annotations,
    normalize_deskew_angle,
)


def test_normalize_deskew_angle_wraps_to_shortest_rotation():
    assert normalize_deskew_angle(170.0) == -10.0
    assert normalize_deskew_angle(-170.0) == 10.0


def test_estimate_skew_from_horizontal_text():
    annotations = [
        {"description": "full"},
        {
            "description": "1007712481",
            "boundingPoly": {"vertices": [{"x": 0, "y": 10}, {"x": 100, "y": 10}, {"x": 100, "y": 30}, {"x": 0, "y": 30}]},
        },
    ]
    assert estimate_skew_degrees_from_annotations(annotations) == 0.0


def test_estimate_skew_from_tilted_text():
    annotations = [
        {"description": "full"},
        {
            "description": "1007712481",
            "boundingPoly": {"vertices": [{"x": 0, "y": 0}, {"x": 100, "y": 10}, {"x": 100, "y": 30}, {"x": 0, "y": 20}]},
        },
    ]
    skew = estimate_skew_degrees_from_annotations(annotations)
    assert skew is not None
    assert 4.0 < skew < 8.0
    assert deskew_rotation_for_baseline(skew) < 0


def test_best_display_angle_prefers_zero_when_skus_match():
    hints = "-10.0:1007712481|-5.0:1007712481|0.0:1007712481|5.0:1007712481"
    assert best_display_angle_from_rotation_hints(hints, "1007712481") == 0.0


def test_best_display_angle_uses_nonzero_when_only_rotated_read_works():
    hints = "-10.0:1007712481|0.0:1007|5.0:1007"
    assert best_display_angle_from_rotation_hints(hints, "1007712481") == -10.0

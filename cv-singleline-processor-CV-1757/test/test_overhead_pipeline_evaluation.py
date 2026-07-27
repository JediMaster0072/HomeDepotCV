from scripts.golden_dataset.evaluate_overhead_pipeline import (
    Box,
    box_iou,
    greedy_one_to_one_matches,
    match_sku_regions_to_tracks,
    polygon_metrics,
    wilson_interval,
)


def _ground_truth(region_key, label, box):
    return {
        "region_key": region_key,
        "label": label,
        "bbox_x1": box[0],
        "bbox_y1": box[1],
        "bbox_x2": box[2],
        "bbox_y2": box[3],
    }


def test_box_iou_uses_union_area():
    assert box_iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == 1 / 3


def test_matching_is_confidence_ordered_class_aware_and_one_to_one():
    ground_truth = [
        _ground_truth("first", "RDC_SKU", (0, 0, 10, 10)),
        _ground_truth("second", "Pallet_SKU", (20, 0, 30, 10)),
    ]
    predictions = [
        {
            "det_id": 0,
            "class_name": "RDC",
            "confidence": 0.8,
            "orig_bbox": {"x1": 1, "y1": 0, "x2": 11, "y2": 10},
        },
        {
            "det_id": 1,
            "class_name": "RDC",
            "confidence": 0.9,
            "orig_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        },
        {
            "det_id": 2,
            "class_name": "RDC",
            "confidence": 0.99,
            "orig_bbox": {"x1": 20, "y1": 0, "x2": 30, "y2": 10},
        },
    ]

    matches, false_positives, false_negatives = greedy_one_to_one_matches(
        predictions,
        ground_truth,
        "orig_bbox",
        0.5,
    )

    assert matches == [{"prediction_index": 1, "gt_index": 0, "iou": 1.0}]
    assert {row["prediction_index"] for row in false_positives} == {0, 2}
    duplicate = next(row for row in false_positives if row["prediction_index"] == 0)
    assert duplicate["best_iou"] > 0.5
    assert false_negatives == [{"gt_index": 1}]


def test_polygon_metrics_report_exact_overlap_and_partial_coverage():
    ground_truth = [[0, 0], [9, 0], [9, 9], [0, 9]]

    exact = polygon_metrics(ground_truth, [ground_truth])
    partial = polygon_metrics(
        ground_truth,
        [[[0, 0], [4, 0], [4, 9], [0, 9]]],
    )

    assert exact == (1.0, 1.0, 1.0)
    assert partial[0] == partial[1] == 0.5
    assert partial[2] == 1.0


def test_wilson_interval_contains_observed_accuracy():
    low, high = wilson_interval(8, 10)

    assert low < 0.8 < high


def test_sku_assignment_uses_center_inside_buffered_crop_and_is_one_to_one():
    ground_truth = [
        _ground_truth("first", "RDC_SKU", (40, 40, 50, 50)),
        _ground_truth("second", "RDC_SKU", (80, 40, 90, 50)),
    ]
    predictions = [
        {
            "det_id": 0,
            "class_name": "RDC",
            "confidence": 0.9,
            "orig_bbox": {"x1": 0, "y1": 0, "x2": 60, "y2": 60},
            "buffered_bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 60},
        },
        {
            "det_id": 1,
            "class_name": "Pallet",
            "confidence": 0.99,
            "orig_bbox": {"x1": 70, "y1": 30, "x2": 100, "y2": 60},
            "buffered_bbox": {"x1": 60, "y1": 20, "x2": 110, "y2": 70},
        },
    ]

    matches, surplus, misses = match_sku_regions_to_tracks(predictions, ground_truth)

    assert len(matches) == 1
    assert matches[0]["prediction_index"] == 0
    assert len(surplus) == 1
    assert len(misses) == 1

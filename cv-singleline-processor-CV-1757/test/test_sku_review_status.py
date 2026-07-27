import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.sku_review import (  # noqa: E402
    is_reviewed_annotation,
    is_scorable_review,
    normalize_review_status,
    parse_expected_sku_input,
    scorability_from_expected_sku,
    sku_digit_class,
)
from common.prediction_suggestions import prediction_suggestions_for_rows  # noqa: E402
from golden_dataset.run_golden_dataset_local import expected_sku_fields  # noqa: E402
from golden_dataset.model_review import (  # noqa: E402
    draw_review_panel,
    load_model_review_status,
    save_model_review_status,
    segmentation_polygons,
)
from golden_dataset.streamlit_expected_sku_review import (  # noqa: E402
    annotation_summary,
    delete_reviewer_and_reassign,
    filter_rows,
    next_region_key_after_save,
    rebalance_reviewer_assignments,
    register_reviewer_and_assign,
    save_annotation,
)


def test_legacy_review_statuses_remain_scorable():
    assert normalize_review_status("reviewed", "1009901192") == "scorable"
    assert normalize_review_status("reviewed_ocr_assisted", "123456") == "scorable"
    assert is_scorable_review("reviewed", "1009901192") is True


def test_legacy_na_rows_receive_a_non_scorable_reason():
    assert normalize_review_status("not_applicable", "N/A", "Bad glare") == "glare"
    assert normalize_review_status("not_applicable", "N/A", "") == "unreadable"
    assert is_reviewed_annotation("not_applicable", "N/A") is True


def test_expected_sku_determines_csv_scorability():
    assert scorability_from_expected_sku("1009901192") == "scorable"
    assert scorability_from_expected_sku("123456") == "scorable"
    assert scorability_from_expected_sku("N/A") == "non-scorable"
    assert scorability_from_expected_sku("") == ""


def test_sku_input_and_digit_class_require_six_or_ten_digits():
    assert parse_expected_sku_input("123456") == ("123456", None)
    assert parse_expected_sku_input("1009901192") == ("1009901192", None)
    assert parse_expected_sku_input("12345")[0] is None
    assert parse_expected_sku_input("X") == ("X", None)
    # X placeholders count toward the 6/10 length (not digits-only).
    assert parse_expected_sku_input("12XX34") == ("12XX34", None)
    assert parse_expected_sku_input("12X456") == ("12X456", None)
    assert parse_expected_sku_input("1X09901192") == ("1X09901192", None)
    assert parse_expected_sku_input("12XX3")[0] is None  # 5 chars after normalize
    assert parse_expected_sku_input("12XX345")[0] is None  # 7 chars
    assert sku_digit_class("123456") == "6-digit"
    assert sku_digit_class("12XX34") == "6-digit"
    assert sku_digit_class("1009901192") == "10-digit"
    assert sku_digit_class("X") == "not-visible"


def test_dynamic_reviewer_assignments_balance_without_splitting_images(tmp_path):
    rows = [
        *[{"image": "many.jpg"} for _ in range(5)],
        *[{"image": "medium.jpg"} for _ in range(3)],
        {"image": "small.jpg"},
    ]
    assignment_path = tmp_path / "assignments.json"
    members, first_assignments = register_reviewer_and_assign(
        assignment_path, "Alice", rows
    )
    assert members == ["Alice"]
    assert set(first_assignments.values()) == {"Alice"}

    members, assignments = register_reviewer_and_assign(
        assignment_path, "Bob", rows
    )

    assert members == ["Alice", "Bob"]
    assert set(assignments) == {"many.jpg", "medium.jpg", "small.jpg"}
    assert all(owner in {"Alice", "Bob"} for owner in assignments.values())
    slot_loads = {
        owner: sum(1 for row in rows if assignments[row["image"]] == owner)
        for owner in members
    }
    assert abs(slot_loads["Alice"] - slot_loads["Bob"]) <= 1

    members, uppercase_assignments = register_reviewer_and_assign(
        assignment_path, "ALICE", rows
    )
    assert members == ["Alice", "Bob"]
    assert uppercase_assignments == assignments

    remaining_members, reassigned = delete_reviewer_and_reassign(
        assignment_path, "aLiCe", rows
    )
    assert remaining_members == ["Bob"]
    assert set(reassigned.values()) == {"Bob"}


def test_started_image_stays_with_its_reviewer_when_new_member_joins():
    rows = [
        {
            "image": "started.jpg",
            "expected_sku": "123456",
            "review_status": "scorable",
            "reviewer": "Alice",
        },
        {"image": "started.jpg", "expected_sku": "", "review_status": ""},
        {"image": "new.jpg", "expected_sku": "", "review_status": ""},
    ]
    assignments = rebalance_reviewer_assignments(rows, ["Alice", "Bob"])

    assert assignments["started.jpg"] == "Alice"
    after_alice_is_removed = rebalance_reviewer_assignments(
        rows,
        ["Bob"],
        assignments,
    )
    assert after_alice_is_removed["started.jpg"] == "Bob"


def test_save_annotation_merges_only_the_selected_row(tmp_path):
    truth_csv = tmp_path / "truth.csv"
    fieldnames = ["region_key", "image", "expected_sku", "reviewer"]
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"region_key": "one", "image": "a.jpg", "expected_sku": "", "reviewer": ""},
                {"region_key": "two", "image": "b.jpg", "expected_sku": "123456", "reviewer": "peer"},
            ]
        )

    rows, saved_fields = save_annotation(
        truth_csv,
        "one",
        {"expected_sku": "1009901192", "reviewer": "Avinash", "sku_digit_class": "10-digit"},
    )

    assert rows[0]["expected_sku"] == "1009901192"
    assert rows[1]["expected_sku"] == "123456"
    assert rows[1]["reviewer"] == "peer"
    assert "sku_digit_class" in saved_fields


def test_bare_predictions_json_infers_image_and_maps_ocr_by_bbox(tmp_path):
    rows = [
        {
            "region_key": "a-one",
            "image": "a.jpg",
            "label": "RDC_SKU",
            "bbox_x1": "10",
            "bbox_y1": "10",
            "bbox_x2": "110",
            "bbox_y2": "60",
        },
        {
            "region_key": "a-two",
            "image": "a.jpg",
            "label": "RDC_SKU",
            "bbox_x1": "200",
            "bbox_y1": "100",
            "bbox_x2": "300",
            "bbox_y2": "150",
        },
        {
            "region_key": "b-one",
            "image": "b.jpg",
            "label": "RDC_SKU",
            "bbox_x1": "500",
            "bbox_y1": "500",
            "bbox_x2": "600",
            "bbox_y2": "550",
        },
    ]
    predictions = [
        {
            "det_id": 1,
            "class_name": "RDC",
            "confidence": 0.9,
            "orig_bbox": {"x1": 10, "y1": 10, "x2": 110, "y2": 60},
            "ocr_words": [{"text": "1002883543"}],
        },
        {
            "det_id": 2,
            "class_name": "RDC",
            "confidence": 0.8,
            "orig_bbox": {"x1": 200, "y1": 100, "x2": 300, "y2": 150},
            "ocr_words": [],
        },
    ]
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")

    represented_images, hints = prediction_suggestions_for_rows(
        prediction_path,
        rows,
    )

    assert represented_images == {"a.jpg"}
    assert hints["a-one"]["text"] == "1002883543"
    assert "a-two" not in hints


def test_per_image_prediction_directory_uses_filename_and_new_segmentation_schema(tmp_path):
    rows = [
        {
            "region_key": "image-one",
            "image": "image.jpg",
            "label": "RDC_SKU",
            "bbox_x1": "10",
            "bbox_y1": "10",
            "bbox_x2": "110",
            "bbox_y2": "60",
        }
    ]
    polygon = [[[12, 12], [108, 12], [108, 58], [12, 58]]]
    prediction = {
        "det_id": 1,
        "class_name": "RDC",
        "confidence": 0.9,
        "orig_bbox": {"x1": 10, "y1": 10, "x2": 110, "y2": 60},
        "buffered_bbox": {"x1": 5, "y1": 5, "x2": 115, "y2": 65},
        "ocr_words": [{"text": "1002883543"}],
        "segmentation": {
            "raw_prediction": {"original_image": {"polygons": polygon}},
            "postprocessed_minAreaRect": {
                "original_image": {"polygons": polygon}
            },
        },
    }
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "image.json").write_text(
        json.dumps([prediction]),
        encoding="utf-8",
    )

    represented_images, hints = prediction_suggestions_for_rows(
        prediction_dir,
        rows,
    )

    assert represented_images == {"image.jpg"}
    assert hints["image-one"]["text"] == "1002883543"
    assert segmentation_polygons(prediction, "raw_prediction") == polygon
    panel = draw_review_panel(
        np.zeros((100, 150, 3), dtype=np.uint8),
        (0, 0, 150, 100),
        polygons=polygon,
    )
    assert panel.shape == (100, 150, 3)
    assert panel.any()


def test_model_review_completion_status_persists(tmp_path):
    status_path = tmp_path / "model_review_status.json"
    save_model_review_status(status_path, {"b.jpg", "a.jpg"})

    assert load_model_review_status(status_path) == {"a.jpg", "b.jpg"}


def test_summary_scores_only_scorable_rows():
    rows = [
        {
            "expected_sku": "1009901192",
            "ocr_crop_suggestion": "1009901192",
            "review_status": "scorable",
        },
        {
            "expected_sku": "123456",
            "ocr_crop_suggestion": "123458",
            "review_status": "reviewed",
        },
        {
            "expected_sku": "N/A",
            "ocr_crop_suggestion": "1000000000",
            "review_status": "motion_blur",
        },
        {"expected_sku": "", "ocr_crop_suggestion": "", "review_status": ""},
    ]

    summary = annotation_summary(rows)

    assert summary == {
        "total": 4,
        "reviewed": 3,
        "scorable": 2,
        "non_scorable": 1,
        "correct": 1,
        "accuracy": 0.5,
    }


def test_annotation_filter_separates_review_states():
    rows = [
        {"expected_sku": "1009901192", "review_status": "scorable"},
        {"expected_sku": "N/A", "review_status": "glare"},
        {"expected_sku": "", "review_status": ""},
    ]

    assert filter_rows(rows, "All", "All", "Scorable") == [rows[0]]
    assert filter_rows(rows, "All", "All", "Non-scorable") == [rows[1]]
    assert filter_rows(rows, "All", "All", "Unreviewed") == [rows[2]]


def test_save_navigation_prefers_next_crop_in_same_image():
    rows = [
        {"region_key": "a-1", "image": "a.jpg"},
        {"region_key": "b-1", "image": "b.jpg"},
        {"region_key": "a-2", "image": "a.jpg"},
    ]

    assert next_region_key_after_save(rows, 0) == "a-2"
    assert next_region_key_after_save(rows, 2) == ""


def test_save_navigation_uses_next_image_after_last_crop():
    rows = [
        {"region_key": "a-1", "image": "a.jpg"},
        {"region_key": "b-1", "image": "b.jpg"},
    ]

    assert next_region_key_after_save(rows, 0) == "b-1"


def test_unreviewed_save_navigation_wraps_to_remaining_crop_in_same_image():
    rows = [
        {"region_key": "a-1", "image": "a.jpg"},
        {"region_key": "a-2", "image": "a.jpg"},
        {"region_key": "b-1", "image": "b.jpg"},
    ]

    assert next_region_key_after_save(rows, 1, wrap_same_image=True) == "a-1"
    assert next_region_key_after_save(rows, 1, wrap_same_image=False) == "b-1"


def test_accuracy_fields_exclude_non_scorable_shape():
    fields = expected_sku_fields(
        {
            "expected_sku": "N/A",
            "expected_sku_review_status": "resolution_too_low",
            "expected_sku_notes": "Digits cannot be resolved",
        },
        [{"text": "1009901192"}],
    )

    assert fields["review_status"] == "resolution_too_low"
    assert fields["accuracy_status"] == "not_applicable"
    assert fields["ocr_match"] == "n/a"

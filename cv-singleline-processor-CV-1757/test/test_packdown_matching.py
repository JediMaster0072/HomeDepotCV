from pipeline.packdown_matching import (
    build_packdown_candidates,
    build_store_inventory_index,
    inventory_observations_from_payload,
    missing_sku_requests_from_empty_grid,
)


def _inventory_payload(store="0244", aisle="09", bay="017"):
    return {
        "store_number": store,
        "aisle_number": aisle,
        "bay_number": bay,
        "photo_timestamp": "1770339128729",
        "photo_location_path": "gs://inventory/box.jpg",
        "process_source": "singleline",
        "inventory_observations": [
            {
                "sku": "1000000456",
                "bbox": [90, 10, 150, 50],
                "confidence": 0.98,
                "class_name": "RDC",
                "source": "seg+google_ocr",
            }
        ],
    }


def test_inventory_observation_keeps_box_location_separate_from_selling_location():
    observations = inventory_observations_from_payload(_inventory_payload())

    assert observations == [
        {
            "store_number": "0244",
            "inventory_aisle": "09",
            "inventory_bay": "017",
            "photo_timestamp": "1770339128729",
            "photo_location_path": "gs://inventory/box.jpg",
            "filename": None,
            "process_source": "singleline",
            "sku": "1000000456",
            "bbox": [90, 10, 150, 50],
            "confidence": 0.98,
            "class_name": "RDC",
            "source": "seg+google_ocr",
            "inventory_bbox": [90, 10, 150, 50],
        }
    ]


def test_missing_selling_sku_matches_inventory_anywhere_in_same_store():
    empty_grid_info = [
        [
            ("", "", 0.0, 0.0, 0, 0.0),
            ("Empty", "1000000456", 19.97, 19.97, 12, 0.91),
        ]
    ]
    requests = missing_sku_requests_from_empty_grid(
        empty_grid_info,
        store_number="244",
        selling_aisle="08",
        selling_bay="003",
    )
    inventory_index = build_store_inventory_index([_inventory_payload()])

    results = build_packdown_candidates(requests, inventory_index)

    assert requests[0]["selling_aisle"] == "08"
    assert requests[0]["selling_bay"] == "003"
    assert results[0]["status"] == "INVENTORY_FOUND"
    assert results[0]["inventory_matches"][0]["inventory_aisle"] == "09"
    assert results[0]["inventory_matches"][0]["inventory_bay"] == "017"


def test_inventory_from_another_store_is_not_returned():
    request = {"sku": "1000000456", "store_number": "0244"}
    inventory_index = build_store_inventory_index(
        [_inventory_payload(store="0121")]
    )

    results = build_packdown_candidates([request], inventory_index)

    assert results[0]["status"] == "NO_INVENTORY_MATCH"
    assert results[0]["inventory_matches"] == []


def test_legacy_bounding_box_payload_remains_searchable():
    payload = {
        "store_number": "0244",
        "aisle_number": "11",
        "bay_number": "002",
        "bounding_boxes": {
            "0000359253": "{'10', '20', '100', '50'}",
        },
    }

    observations = inventory_observations_from_payload(payload)

    assert observations[0]["sku"] == "0000359253"
    assert observations[0]["inventory_bbox"] == [10.0, 20.0, 100.0, 50.0]

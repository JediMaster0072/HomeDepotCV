from utils.google_ocr_utils import (
    is_strong_sku_read,
    merge_multi_pass_sku_results,
    primary_sku_suggestion,
)


def test_is_strong_sku_read():
    assert is_strong_sku_read([{"text": "1004815592"}])
    assert is_strong_sku_read([{"text": "482888"}])
    assert not is_strong_sku_read([])
    assert not is_strong_sku_read([{"text": "abc"}])


def test_merge_multi_pass_prefers_ten_digit_and_upright():
    upright = [{"text": "1004815592", "source": "google_ocr_sku_parse"}]
    rotated = [{"text": "1004815593", "source": "google_ocr_sku_parse_rotneg5_retry"}]
    merged = merge_multi_pass_sku_results([upright, rotated])
    texts = {item["text"] for item in merged}
    assert texts == {"1004815592", "1004815593"}
    assert primary_sku_suggestion(merged) == "1004815592"


def test_merge_multi_pass_keeps_best_source_for_same_sku():
    upright = [{"text": "1004815592", "source": "google_ocr_sku_parse"}]
    rotated = [{"text": "1004815592", "source": "google_ocr_sku_parse_rot10_retry"}]
    merged = merge_multi_pass_sku_results([upright, rotated])
    assert len(merged) == 1
    assert merged[0]["source"] == "google_ocr_sku_parse"

# Overhead Pipeline Evaluation

- Images evaluated: 107
- SKU ground-truth regions: 1410
- OD ground-truth regions: 1445
- Predictions: 3165
- IoU threshold: 0.5

## Summary metrics

- **images_evaluated**: 107
- **sku_ground_truth_regions**: 1410
- **od_ground_truth_regions**: 1445
- **predictions**: 3165
- **empty_prediction_images**: 0
- **duplicate_det_ids**: 0
- **segmentation_found_rate**: 0.8619
- **ocr_found_rate**: 0.3030
- **valid_6_or_10_digit_sku_rate**: 0.3030
- **duplicate_overlapping_predictions**: 0
- **detection_precision_at_iou_0.5**: 0.3918
- **detection_recall_at_iou_0.5**: 0.8581
- **detection_f1_at_iou_0.5**: 0.5380
- **od_raw_mean_matched_iou**: 0.8850
- **od_raw_mean_gt_coverage**: 0.9391
- **od_raw_full_gt_coverage_rate**: 0.1347
- **od_buffered_mean_matched_iou**: 0.6031
- **od_buffered_mean_gt_coverage**: 0.9958
- **od_buffered_full_gt_coverage_rate**: 0.9460
- **interim_end_to_end_ocr_exact_accuracy**: 0.7782
- **interim_ocr_accuracy_ci95_low**: 0.7214
- **interim_ocr_accuracy_ci95_high**: 0.8263
- **interim_conditional_ocr_exact_accuracy**: 0.9442
- **scorable_any_track_center_coverage_rate**: 0.9665
- **scorable_sku_track_match_rate**: 0.8619
- **scorable_buffered_full_coverage_rate**: 0.9709
- **scorable_buffered_mean_gt_coverage**: 0.9971
- **raw_segmentation_mean_iou_proxy**: 0.6604
- **raw_segmentation_mean_gt_coverage**: 0.8299
- **post_segmentation_mean_iou_proxy**: 0.3844
- **post_segmentation_mean_gt_coverage**: 0.9430

## Failure categories

- **correct**: 183
- **detection_missed**: 213
- **not_ocr_scorable**: 896
- **ocr_incorrect**: 11
- **ocr_missing**: 8
- **poor_crop_coverage**: 70
- **segmentation_missing**: 29

## Detection-level failures

- **detection_missed**: 205
- **false_positive**: 1925

## Notes

- Segmentation metrics are region proxies unless the ground-truth polygons are confirmed masks.
- OCR metrics are interim and automatically update as more expected SKUs are reviewed.
- OD matches parent-object GT by raw-box IoU; SKU stages match label centers inside buffered crops.
- Both matching planes are class-aware and one-to-one.
- Any-track center coverage is also reported before one-to-one collision resolution.

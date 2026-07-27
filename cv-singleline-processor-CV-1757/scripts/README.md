# Scripts

Local tooling for golden-dataset OCR evaluation, empty-shelf audit, and docs maintenance.

## Layout

| Folder | Purpose |
|--------|---------|
| `common/` | Shared path/bootstrap helpers |
| `golden_dataset/` | Golden overhead dataset OCR, overlays, expected_sku review |
| `empty_shelf/` | camera_cart EmptyItem review and temporal baseline metrics |
| `docs/` | Regenerate markdown table-of-contents files |

## Setup

From the repo root:

```bash
python -m pip install -r scripts/requirements-tools.txt
```

## Golden dataset

```bash
python scripts/golden_dataset/apply_drona_jsons.py
python scripts/golden_dataset/generate_golden_sku_truth_csv.py --run-crops
python scripts/golden_dataset/run_golden_dataset_local.py --mode inventory
python scripts/golden_dataset/generate_golden_label_overlays.py
streamlit run scripts/golden_dataset/streamlit_expected_sku_review.py
python scripts/golden_dataset/sync_expected_sku_from_truth_csv.py
```

## Empty shelf

```bash
python scripts/empty_shelf/generate_empty_region_truth_csv.py
streamlit run scripts/empty_shelf/streamlit_empty_region_review.py
python scripts/empty_shelf/evaluate_empty_shelf_detector_audit.py
python scripts/empty_shelf/evaluate_empty_shelf_temporal_baseline.py
```

## Docs helpers

```bash
python scripts/docs/update_local_testing_guide_toc.py
python scripts/docs/update_docs_readme_toc.py
```

## Remote Streamlit (5090 host)

```bash
./scripts/deploy/check_gpu_host_access.sh
./scripts/deploy/deploy_streamlit_to_gpu_host.sh
```

See `scripts/deploy/README.md` for team access URL and post-upload steps.

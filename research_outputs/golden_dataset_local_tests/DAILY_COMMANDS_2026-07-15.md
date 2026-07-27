# Golden Dataset / SKU Review — Commands Used (2026-07-15)

Quick reference for commands run today during golden-dataset labeling, OCR, rotation eval, Streamlit review, and 5090 deploy/sync.

**Paths**

| Item | Location |
|------|----------|
| Project root | `~/Downloads/HomeDepotCV` |
| Processor repo | `~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757` |
| Python venv | `~/Downloads/HomeDepotCV/.venv` |
| Truth CSV | `research_outputs/golden_dataset_local_tests/golden_sku_truth.csv` |
| Dataset (expected SKU) | `Golden_Dataset_overhead_eval_expected_sku/` |
| Team Streamlit (VPN) | http://172.16.20.108:8503 |
| Local Streamlit | http://localhost:8501 |
| SSH key | `~/Downloads/HomeDepotCV/avinash_patel_lf.pem` |

**Note:** macOS often has no `python` on PATH — use the venv below.

---

## 1. Environment setup

```bash
# Activate venv (recommended)
source ~/Downloads/HomeDepotCV/.venv/bin/activate

# Or call Python directly
~/Downloads/HomeDepotCV/.venv/bin/python --version

# Install Streamlit review deps (if needed)
pip install streamlit pandas pillow opencv-python-headless
```

**OCR credentials** — put in `~/Downloads/HomeDepotCV/.env`:

```bash
GOOGLE_OCR_API_KEY=<your_key>
# optional:
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

```bash
# Restrict permissions on secrets file
chmod 600 ~/.home_depot_cv.env
```

**gcloud ADC (alternative to API key):**

```bash
gcloud auth application-default login
gcloud config set project <your-gcp-project-id>
gcloud auth application-default set-quota-project <your-gcp-project-id>
```

---

## 2. Working directory

Most commands assume:

```bash
cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757
```

---

## 3. Drona JSON → dataset → truth CSV pipeline

Apply updated Drona annotations, regenerate crops/overlays, rebuild truth CSV:

```bash
source ~/Downloads/HomeDepotCV/.venv/bin/activate
cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757

# Dry-run first (optional)
python scripts/golden_dataset/apply_drona_jsons.py --dry-run

# Apply Drona JSONs to expected-SKU dataset copy
python scripts/golden_dataset/apply_drona_jsons.py

# Full pipeline (what we ran several times today)
python scripts/golden_dataset/apply_drona_jsons.py && \
python scripts/golden_dataset/run_golden_dataset_local.py --mode crops --save-crops --min-crop-short-side 720 && \
python scripts/golden_dataset/generate_golden_label_overlays.py && \
python scripts/golden_dataset/generate_golden_sku_truth_csv.py --force
```

**Regenerate truth CSV only** (preserve labels from prior CSV):

```bash
python scripts/golden_dataset/generate_golden_sku_truth_csv.py --force \
  --prior-truth-csv ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.5090_latest.csv
```

**Generate overlays only:**

```bash
python scripts/golden_dataset/generate_golden_label_overlays.py \
  --dataset ~/Downloads/HomeDepotCV/Golden_Dataset_overhead_eval_expected_sku \
  --output-dir ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/label_overlays_expected_sku
```

---

## 4. Golden dataset runner (`run_golden_dataset_local.py`)

**Inventory:**

```bash
~/Downloads/HomeDepotCV/.venv/bin/python scripts/golden_dataset/run_golden_dataset_local.py --mode inventory
```

**Save HD upscaled crops (720px short side):**

```bash
# Option A: venv activated
python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode crops --save-crops --min-crop-short-side 720

# Option B: full path
~/Downloads/HomeDepotCV/.venv/bin/python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode crops --save-crops --min-crop-short-side 720
```

**OCR on crops (upright / 0°):**

```bash
python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode ocr-crops --save-crops \
  --ocr-auth api-key \
  --env-file ~/Downloads/HomeDepotCV/.env

# Smoke test (5 images)
python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode ocr-crops --limit 5 --save-crops \
  --ocr-auth api-key \
  --env-file ~/Downloads/HomeDepotCV/.env
```

**Rotation crop variants (no OCR):**

```bash
python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode rotation-crops --limit 1 \
  --rotation-angles 0,180,-10,10,-5,5
```

**Full rotation OCR run (1,410 crops × 6 angles ≈ 8,460 API calls — hours):**

```bash
# Foreground
python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode ocr-rotation-crops \
  --rotation-angles 0,180,-10,10,-5,5 \
  --ocr-auth api-key \
  --env-file ~/Downloads/HomeDepotCV/.env

# Background (what we used for the long run)
nohup python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode ocr-rotation-crops \
  --rotation-angles 0,180,-10,10,-5,5 \
  --ocr-auth api-key \
  --env-file ~/Downloads/HomeDepotCV/.env \
  > ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/ocr_rotation_run.log 2>&1 &

tail -f ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/ocr_rotation_run.log
```

**Monitor rotation OCR progress** (count `.jpg` files created today; divide by 6 ≈ crops done):

```bash
find ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/rotation_crops \
  -name "*.jpg" -newermt "2026-07-15 13:00" | wc -l
# target: 8460 files (= 1410 crops × 6 angles)
```

---

## 5. Label sync + OCR accuracy

After labeling in Streamlit, sync CSV → JSON, then re-run OCR to measure accuracy:

```bash
python scripts/golden_dataset/sync_expected_sku_from_truth_csv.py

python scripts/golden_dataset/run_golden_dataset_local.py \
  --mode ocr-crops --save-crops \
  --ocr-auth api-key \
  --env-file ~/Downloads/HomeDepotCV/.env
```

Accuracy summary prints at end of `ocr-crops` run (uses `expected_sku` from JSON).

**Normalize dash-formatted SKUs on 5090** (if needed):

```bash
cd ~/HomeDepotCV/cv-singleline-processor-CV-1757
source ~/HomeDepotCV/.venv/bin/activate
python scripts/golden_dataset/normalize_expected_sku_dashes.py
```

---

## 6. Streamlit SKU review

**Local (port 8501):**

```bash
cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757
./scripts/golden_dataset/start_streamlit_review.sh

# Stop local only
./scripts/golden_dataset/start_streamlit_review.sh --kill

# Manual start
streamlit run scripts/golden_dataset/streamlit_expected_sku_review.py
```

**Review batch filter (added today):**

- Sidebar → **Review scope → Current review batch** → ~341 rows (46 images), ~187 unreviewed
- Batch manifest: `research_outputs/golden_dataset_local_tests/review_batch_images.txt`
- Use **All images** for the full 1,410-row queue

---

## 7. 5090 GPU host — deploy & sync

**Check VPN / SSH access:**

```bash
cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757
./scripts/deploy/check_gpu_host_access.sh

ssh -i ~/Downloads/HomeDepotCV/avinash_patel_lf.pem avinash.patel@172.16.20.108
```

**First-time deploy:**

```bash
./scripts/deploy/deploy_streamlit_to_gpu_host.sh
```

**Update code + data + restart (⚠️ OVERWRITES remote `golden_sku_truth.csv` with Mac copy):**

```bash
./scripts/deploy/deploy_streamlit_to_gpu_host.sh --update-only
```

**Update code only — keep remote labels (safe for UI/crop changes):**

```bash
./scripts/deploy/deploy_streamlit_to_gpu_host.sh --update-only --code-only
```

**Push corrected labels to 5090 after restoring from backup:**

```bash
cp ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.5090_latest.csv \
   ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv

cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757
./scripts/deploy/deploy_streamlit_to_gpu_host.sh --update-only
```

**Snapshot Mac CSV before deploy:**

```bash
cp ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv \
   ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.5090_latest.csv
```

**Pull team labels from 5090 → Mac:**

```bash
cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757

# Replace local (backs up first)
./scripts/deploy/pull_annotations_from_gpu_host.sh

# Merge remote into local without overwriting existing labels
./scripts/deploy/pull_annotations_from_gpu_host.sh --merge
```

**Manual rsync pull:**

```bash
rsync -az -e "ssh -i ~/Downloads/HomeDepotCV/avinash_patel_lf.pem" \
  avinash.patel@172.16.20.108:~/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv \
  ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.remote_5090.csv
```

**5090 Streamlit logs:**

```bash
ssh -i ~/Downloads/HomeDepotCV/avinash_patel_lf.pem avinash.patel@172.16.20.108 \
  'sudo journalctl -u streamlit-golden-sku -f'

# Or nohup log
ssh avinash.patel@172.16.20.108 'tail -f ~/HomeDepotCV/logs/streamlit-golden-sku.log'
```

**Manual 5090 Streamlit restart (if systemd unavailable):**

```bash
ssh avinash.patel@172.16.20.108 \
  'pkill -f streamlit_expected_sku_review; cd ~/HomeDepotCV/cv-singleline-processor-CV-1757 && nohup ~/HomeDepotCV/.venv/bin/streamlit run scripts/golden_dataset/streamlit_expected_sku_review.py --server.address 0.0.0.0 --server.port 8503 --server.headless true > ~/HomeDepotCV/logs/streamlit-golden-sku.log 2>&1 &'
```

### Recommended team workflow

```bash
# Before pushing to team:
./scripts/deploy/pull_annotations_from_gpu_host.sh --merge

# Code/data push (or --code-only to preserve remote labels):
./scripts/deploy/deploy_streamlit_to_gpu_host.sh --update-only --code-only

# End of day — collect peer labels:
./scripts/deploy/pull_annotations_from_gpu_host.sh --merge
```

---

## 8. Truth CSV inspection

**Count labeled rows (Mac or 5090):**

```bash
cd ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests

wc -l golden_sku_truth.csv
ls -la golden_sku_truth*.csv

python3 -c "
import csv
rows = list(csv.DictReader(open('golden_sku_truth.csv')))
labeled = sum(1 for r in rows if (r.get('expected_sku') or '').strip())
print(len(rows), 'rows,', labeled, 'with expected_sku,', len(rows)-labeled, 'unreviewed')
"
```

**Diff local vs remote:**

```bash
diff research_outputs/golden_dataset_local_tests/golden_sku_truth.csv \
     research_outputs/golden_dataset_local_tests/golden_sku_truth.remote_5090.csv
```

**Key CSV copies from today:**

| File | Purpose |
|------|---------|
| `golden_sku_truth.csv` | Live working copy |
| `golden_sku_truth.5090_latest.csv` | Last known good 5090 snapshot (154 labels) |
| `golden_sku_truth.labeled_subset_5090.csv` | Rows with `expected_sku` filled |
| `golden_sku_truth.backup_20260715_095916.csv` | Pre-restore backup (84 labels) |
| `ocr-accuracy_labeled_subset_5090_summary.csv` | Accuracy run output |
| `ocr-rotation-crops_summary.csv` | Full rotation OCR results (8,460 rows) |

---

## 9. Open / inspect crops

```bash
open ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/crops/1770339044281_0244_1031_07-020/000_RDC_SKU_330_1904_608_2010.jpg

open ~/Downloads/HomeDepotCV/research_outputs/golden_dataset_local_tests/crops/1770339128729_0244_1163_08-009/004_RDC_SKU_2687_590_2813_631.jpg
```

---

## 10. Tests (multi-angle OCR)

```bash
cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757
source ~/Downloads/HomeDepotCV/.venv/bin/activate

python -m pytest test/test_ocr_multi_angle.py -v
```

---

## 11. GPU / Docker (detector smoke test — in progress)

```bash
ssh -i ~/Downloads/HomeDepotCV/avinash_patel_lf.pem avinash.patel@172.16.20.108

nvidia-smi

cd ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
sudo docker build -t hd-singleline-det-current:local .
```

---

## 12. Deploy after today's batch-filter fix

When VPN is connected:

```bash
cd ~/Downloads/HomeDepotCV/cv-singleline-processor-CV-1757
./scripts/deploy/deploy_streamlit_to_gpu_host.sh --update-only --code-only
```

Then hard-refresh http://172.16.20.108:8503 and set **Review scope → Current review batch**.

---

## 13. Safety reminders

| Action | Risk |
|--------|------|
| `deploy_streamlit_to_gpu_host.sh --update-only` (without `--code-only`) | **Overwrites** remote `golden_sku_truth.csv` with Mac copy |
| Labeling on both localhost:8501 and 5090:8503 at once | CSVs drift out of sync |
| `pull_annotations --merge` | Only fills **empty** local rows; does not overwrite your labels |

**Always `--code-only`** when deploying UI/crop improvements but peers are actively labeling on 5090.

**Always `pull --merge` first** before a full `--update-only` push if peers may have added labels.

#!/usr/bin/env bash
# Deploy the golden SKU Streamlit review app to the shared RTX 5090 host.
#
# Prereqs (on your Mac):
#   - LF/VPN connected so 172.16.20.108 is reachable
#   - SSH key at avinash_patel_lf.pem (or set HOME_DEPOT_SSH_KEY)
#
# Usage:
#   ./scripts/deploy/deploy_streamlit_to_gpu_host.sh                 # first-time setup
#   ./scripts/deploy/deploy_streamlit_to_gpu_host.sh --update-only   # push local code+data, restart 8503
#   ./scripts/deploy/deploy_streamlit_to_gpu_host.sh --update-only --code-only  # push code only (keep remote CSV)
#   ./scripts/deploy/pull_annotations_from_gpu_host.sh --merge      # pull peer labels to your Mac first
#
# Annotation storage:
#   Local Streamlit (8501) saves to:
#     HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv
#   Team Streamlit on 5090 (8503) saves to the SAME PATH on the GPU host — NOT your Mac.
#   Workflow: pull --merge after peers review, then push --update-only when you change code/data.
#
# The 5090 host runs Streamlit on port 8503 (systemd: streamlit-golden-sku.service).
# Local dev stays on http://localhost:8501 — do not use 8503 locally.
#
# Team URL (VPN required): http://172.16.20.108:8503
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

HOST="172.16.20.108"
USER="${HOME_DEPOT_SSH_USER:-avinash.patel}"
KEY="${HOME_DEPOT_SSH_KEY:-$PROJECT_ROOT/avinash_patel_lf.pem}"
PORT=8503
UPDATE_ONLY=0
CODE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --update-only) UPDATE_ONLY=1; shift ;;
    --code-only) CODE_ONLY=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no "${USER}@${HOST}")
RSYNC=(rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no")

echo "Deploy target: ${USER}@${HOST}"
echo "Local project root: $PROJECT_ROOT"
echo

"${SSH[@]}" "mkdir -p ~/HomeDepotCV/research_outputs/golden_dataset_local_tests/crops \
  ~/HomeDepotCV/research_outputs/golden_dataset_local_tests/label_overlays_expected_sku \
  ~/HomeDepotCV/Golden_Dataset_overhead_eval_expected_sku"

echo "Syncing processor repo scripts..."
"${RSYNC[@]}" \
  --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  "$REPO_ROOT/scripts/" \
  "${USER}@${HOST}:~/HomeDepotCV/cv-singleline-processor-CV-1757/scripts/"

if [[ -f "$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/review_batch_images.txt" ]]; then
  echo "Syncing review batch manifest..."
  "${RSYNC[@]}" \
    "$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/review_batch_images.txt" \
    "${USER}@${HOST}:~/HomeDepotCV/research_outputs/golden_dataset_local_tests/"
fi

if [[ -f "$REPO_ROOT/utils/crop_preprocess.py" ]]; then
  echo "Syncing utils/crop_preprocess.py..."
  "${RSYNC[@]}" \
    "$REPO_ROOT/utils/crop_preprocess.py" \
    "${USER}@${HOST}:~/HomeDepotCV/cv-singleline-processor-CV-1757/utils/"
fi

if [[ "$UPDATE_ONLY" -eq 0 ]]; then
  echo "Syncing requirements..."
  "${RSYNC[@]}" \
    "$REPO_ROOT/requirements.txt" \
    "$REPO_ROOT/scripts/requirements-tools.txt" \
    "${USER}@${HOST}:~/HomeDepotCV/cv-singleline-processor-CV-1757/"

  "${RSYNC[@]}" \
    "$REPO_ROOT/scripts/requirements-tools.txt" \
    "${USER}@${HOST}:~/HomeDepotCV/cv-singleline-processor-CV-1757/scripts/"
fi

echo "Syncing research outputs (truth CSV, OCR summary, crops, overlays)..."
if [[ "$CODE_ONLY" -eq 0 ]]; then
  "${RSYNC[@]}" \
    "$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv" \
    "$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/ocr-crops_summary.csv" \
    "$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/review_batch_images.txt" \
    "${USER}@${HOST}:~/HomeDepotCV/research_outputs/golden_dataset_local_tests/" 2>/dev/null || true

  "${RSYNC[@]}" \
    "$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/crops/" \
    "${USER}@${HOST}:~/HomeDepotCV/research_outputs/golden_dataset_local_tests/crops/" 2>/dev/null || true

  "${RSYNC[@]}" \
    "$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/label_overlays_expected_sku/" \
    "${USER}@${HOST}:~/HomeDepotCV/research_outputs/golden_dataset_local_tests/label_overlays_expected_sku/" 2>/dev/null || true
else
  echo "Code-only mode: skipping truth CSV / crops / overlays (remote annotations preserved)."
fi

if [[ "$UPDATE_ONLY" -eq 1 ]]; then
  echo "Update-only: restarting existing streamlit-golden-sku service on port ${PORT}..."
  "${SSH[@]}" bash -s "$PORT" <<'REMOTE'
set -euo pipefail
PORT="$1"
sudo -n systemctl restart streamlit-golden-sku.service
sleep 2
sudo -n systemctl is-active streamlit-golden-sku.service
curl -s -o /dev/null -w "HTTP %{http_code}\n" "http://127.0.0.1:${PORT}"
wc -l ~/HomeDepotCV/research_outputs/golden_dataset_local_tests/golden_sku_truth.csv
REMOTE
  echo
  echo "Update complete."
  echo "Team URL (VPN required): http://${HOST}:${PORT}"
  exit 0
fi

echo "Remote setup: venv, packages, systemd service..."
"${SSH[@]}" bash -s "$PORT" <<'REMOTE'
set -euo pipefail
PORT="$1"
cd ~/HomeDepotCV

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -U pip -q
pip install -r cv-singleline-processor-CV-1757/scripts/requirements-tools.txt -q

SERVICE_FILE=cv-singleline-processor-CV-1757/scripts/deploy/streamlit-golden-sku.service
sed "s/--server.port 8501/--server.port ${PORT}/" "$SERVICE_FILE" > /tmp/streamlit-golden-sku.service

if sudo -n cp /tmp/streamlit-golden-sku.service /etc/systemd/system/streamlit-golden-sku.service 2>/dev/null; then
  sudo -n systemctl daemon-reload
  sudo -n systemctl enable streamlit-golden-sku.service
  sudo -n systemctl restart streamlit-golden-sku.service
  sleep 2
  sudo -n systemctl --no-pager status streamlit-golden-sku.service | head -15
else
  echo "sudo unavailable — starting Streamlit via nohup on port ${PORT}..."
  pkill -f "streamlit run scripts/golden_dataset/streamlit_expected_sku_review.py" 2>/dev/null || true
  mkdir -p ~/HomeDepotCV/logs
  cd ~/HomeDepotCV/cv-singleline-processor-CV-1757
  nohup ~/HomeDepotCV/.venv/bin/streamlit run scripts/golden_dataset/streamlit_expected_sku_review.py \
    --server.address 0.0.0.0 --server.port "${PORT}" \
    --server.headless true \
    > ~/HomeDepotCV/logs/streamlit-golden-sku.log 2>&1 &
  sleep 3
  pgrep -af streamlit_expected_sku_review || true
  tail -5 ~/HomeDepotCV/logs/streamlit-golden-sku.log || true
fi
REMOTE

echo
echo "Deploy complete."
echo "Team URL (VPN required): http://${HOST}:${PORT}"
echo "Logs: ssh ${USER}@${HOST} 'sudo journalctl -u streamlit-golden-sku -f'"

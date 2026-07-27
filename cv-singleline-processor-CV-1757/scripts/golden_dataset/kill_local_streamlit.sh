#!/usr/bin/env bash
# Stop local Streamlit processes on this machine only.
#
# Does NOT touch the team instance on the 5090 GPU host (172.16.20.108:8503).
# Port 8503 is reserved for that remote systemd service — never killed locally.
#
# Usage:
#   ./scripts/golden_dataset/kill_local_streamlit.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PID_FILE="$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/streamlit.pid"

echo "Stopping local Streamlit processes..."
echo "(5090 team app at http://172.16.20.108:8503 is NOT affected)"

if [[ -f "$PID_FILE" ]]; then
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  rm -f "$PID_FILE"
fi

# Golden SKU review (local)
pkill -f "streamlit run scripts/golden_dataset/streamlit_expected_sku_review.py" 2>/dev/null || true

# Streamlit labeller (local, port 8502)
pkill -f "streamlit run app.py --server.port 8502" 2>/dev/null || true
pkill -f "streamlit_labeller.*streamlit run" 2>/dev/null || true

# Free local review ports only — 8503 is reserved for 5090 remote instance
if command -v lsof >/dev/null 2>&1; then
  for port in 8501 8502; do
    lsof -ti ":${port}" 2>/dev/null | xargs kill -9 2>/dev/null || true
  done
fi

sleep 1
echo "Local Streamlit stopped (ports 8501, 8502). Remote 8503 untouched."

#!/usr/bin/env bash
# Start the golden SKU Streamlit review app locally (port 8501).
#
# Kills other LOCAL Streamlit apps first (8501, 8502).
# Does NOT touch the 5090 team instance (172.16.20.108:8503).
#
# Usage:
#   ./scripts/golden_dataset/start_streamlit_review.sh          # kill local + start
#   ./scripts/golden_dataset/start_streamlit_review.sh --kill   # stop local only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
PORT=8501
LOG="$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/streamlit.log"
PID_FILE="$PROJECT_ROOT/research_outputs/golden_dataset_local_tests/streamlit.pid"
KILL_SCRIPT="$SCRIPT_DIR/kill_local_streamlit.sh"

if [[ "${1:-}" == "--kill" ]]; then
  bash "$KILL_SCRIPT"
  exit 0
fi

if [[ ! -x "$VENV/bin/streamlit" ]]; then
  echo "Missing venv at $VENV"
  exit 1
fi

bash "$KILL_SCRIPT"
mkdir -p "$(dirname "$LOG")"
cd "$REPO_ROOT"

echo "Starting golden SKU label review on port ${PORT}..."
nohup "$VENV/bin/streamlit" run scripts/golden_dataset/streamlit_expected_sku_review.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  >> "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "$PID_FILE"
sleep 4

ok=0
for url in "http://127.0.0.1:${PORT}/_stcore/health" "http://localhost:${PORT}/_stcore/health"; do
  if curl -sf "$url" >/dev/null 2>&1; then
    echo "Health OK: $url"
    ok=1
  else
    echo "Health FAIL: $url"
  fi
done

if [[ "$ok" -eq 1 ]]; then
  echo "Ready:"
  echo "  http://127.0.0.1:${PORT}"
  echo "  http://localhost:${PORT}"
  echo "PID: ${PID}  Log: $LOG"
  echo "5090 team app (untouched): http://172.16.20.108:8503"
else
  echo "Streamlit failed to start — last log lines:"
  tail -30 "$LOG"
  exit 1
fi

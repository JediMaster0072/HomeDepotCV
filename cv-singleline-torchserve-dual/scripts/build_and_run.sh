#!/usr/bin/env bash
# Build hd-dual-gpu from HomeDepotCV repo root.
# Usage:
#   ./cv-singleline-torchserve-dual/scripts/build_and_run.sh [--build-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DUAL_DIR}/.." && pwd)"
IMAGE="${IMAGE:-hd-dual-gpu}"
CONTAINER="${CONTAINER:-hd-dual-gpu}"
HTTP_PORT="${HTTP_PORT:-9000}"
MGMT_PORT="${MGMT_PORT:-9001}"
METRICS_PORT="${METRICS_PORT:-9002}"

DET_DIR="${REPO_ROOT}/cv-singleline-detector-yolo7_det_dep_2"
SEG_DIR="${REPO_ROOT}/cv-singleline-detector-yolov7-seg"

log() { printf '[hd-dual] %s\n' "$*"; }

if [[ ! -f "${DET_DIR}/best.pt" ]]; then
  log "ERROR: missing ${DET_DIR}/best.pt"
  exit 1
fi
if [[ ! -f "${SEG_DIR}/segmentation.pt" ]]; then
  log "ERROR: missing ${SEG_DIR}/segmentation.pt"
  exit 1
fi

log "Building ${IMAGE} (context=${REPO_ROOT})"
docker build \
  -f "${DUAL_DIR}/Dockerfile" \
  -t "${IMAGE}" \
  "${REPO_ROOT}"

if [[ "${1:-}" == "--build-only" ]]; then
  log "Build complete (--build-only)"
  exit 0
fi

log "Stopping old dual / split containers if present"
docker rm -f "${CONTAINER}" hd-det-gpu hd-seg-gpu 2>/dev/null || true

log "Starting ${CONTAINER} on :${HTTP_PORT}/:${MGMT_PORT}/:${METRICS_PORT}"
docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  -p "${HTTP_PORT}:8080" \
  -p "${MGMT_PORT}:8081" \
  -p "${METRICS_PORT}:8082" \
  --shm-size=8g \
  --restart unless-stopped \
  "${IMAGE}"

log "Waiting for Healthy…"
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${HTTP_PORT}/ping" | grep -q Healthy; then
    log "ping Healthy"
    break
  fi
  sleep 5
  if [[ "$i" -eq 60 ]]; then
    log "ERROR: ping timeout — docker logs ${CONTAINER}"
    docker logs --tail 80 "${CONTAINER}" || true
    exit 1
  fi
done

log "Models:"
curl -s "http://127.0.0.1:${MGMT_PORT}/models" | python3 -m json.tool || true
log "Done. Endpoints:"
log "  POST http://127.0.0.1:${HTTP_PORT}/predictions/detector"
log "  POST http://127.0.0.1:${HTTP_PORT}/predictions/segmenter"

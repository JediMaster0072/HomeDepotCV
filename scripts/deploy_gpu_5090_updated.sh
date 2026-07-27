#!/usr/bin/env bash
# Deploy both single-line TorchServe detector services on GPU5-A5090 (172.16.20.108).
#
# Prerequisites on the host:
#   - NVIDIA driver + Docker (--gpus all works; see gpu-docker-cdi-fix.md)
#   - gsutil (or gcloud) for model.sh weight downloads
#   - Repo checked out with both detector directories present
#
# Usage:
#   ./scripts/deploy_gpu_5090.sh              # build + run + smoke test
#   ./scripts/deploy_gpu_5090.sh --build-only
#   ./scripts/deploy_gpu_5090.sh --run-only
#   INFERENCE_IMAGE=/path/to/shelf.jpg ./scripts/deploy_gpu_5090.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
DET_DIR="${REPO_ROOT}/cv-singleline-detector-yolo7_det_dep_2"
SEG_DIR="${REPO_ROOT}/cv-singleline-detector-yolov7-seg"

DET_IMAGE="${DET_IMAGE:-hd-det-gpu}"
SEG_IMAGE="${SEG_IMAGE:-hd-seg-gpu}"
DET_CONTAINER="${DET_CONTAINER:-hd-det-gpu}"
SEG_CONTAINER="${SEG_CONTAINER:-hd-seg-gpu}"

DET_HTTP_PORT="${DET_HTTP_PORT:-9000}"
DET_MGMT_PORT="${DET_MGMT_PORT:-9001}"
SEG_HTTP_PORT="${SEG_HTTP_PORT:-10000}"
SEG_MGMT_PORT="${SEG_MGMT_PORT:-10001}"

DO_BUILD=1
DO_RUN=1

for arg in "$@"; do
  case "${arg}" in
    --build-only) DO_RUN=0 ;;
    --run-only) DO_BUILD=0 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

wait_for_healthy() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  local i=1
  while (( i <= attempts )); do
    if curl -sf "${url}" | grep -q '"status"[[:space:]]*:[[:space:]]*"Healthy"'; then
      log "${label} is healthy (${url})"
      return 0
    fi
    sleep 2
    (( i++ )) || true
  done
  die "${label} did not become healthy at ${url} within $((attempts * 2))s"
}

wait_for_model_ready() {
  local url="$1"
  local label="$2"
  local attempts="${3:-90}"
  local i=1
  while (( i <= attempts )); do
    if curl -sf "${url}" | grep -q '"status"[[:space:]]*:[[:space:]]*"READY"'; then
      log "${label} workers are READY"
      return 0
    fi
    sleep 3
    (( i++ )) || true
  done
  die "${label} workers not READY at ${url}"
}

download_weights() {
  local dir="$1"
  local weight_file="$2"
  log "Downloading weights in ${dir} (expect ${weight_file})"
  (
    cd "${dir}"
    if [[ -f "${weight_file}" ]]; then
      log "  ${weight_file} already present — skipping gsutil"
    else
      require_cmd gsutil
      bash model.sh
    fi
    [[ -f "${weight_file}" ]] || die "Weight file missing after model.sh: ${dir}/${weight_file}"
  )
}

build_images() {
  download_weights "${DET_DIR}" "best.pt"
  download_weights "${SEG_DIR}" "segmentation.pt"

  log "Building detection image: ${DET_IMAGE}"
  docker build -t "${DET_IMAGE}" "${DET_DIR}"

  log "Building segmentation image: ${SEG_IMAGE}"
  docker build -t "${SEG_IMAGE}" "${SEG_DIR}"
}


check_torch_gpu_support() {
  local image="$1"
  local label="$2"

  log "Checking PyTorch GPU support in ${image}"

  docker run --rm --gpus all --entrypoint python3 "${image}" -c '
import torch

assert torch.cuda.is_available(), "CUDA not available"

cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
supported = torch.cuda.get_arch_list()

print("GPU architecture:", arch)
print("Supported:", supported)

assert arch in supported, f"{arch} NOT supported"
'

  if [[ $? -ne 0 ]]; then
      die "${label} image lacks RTX 5090 (sm_120) support."
  fi
}

run_containers() {
  require_cmd docker
  require_cmd curl

  log "GPU preflight (both RTX 5090s visible inside Docker)"
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

  log "Stopping any existing containers: ${DET_CONTAINER}, ${SEG_CONTAINER}"
  docker rm -f "${DET_CONTAINER}" "${SEG_CONTAINER}" 2>/dev/null || true

  log "Starting detection on GPU 0 → localhost:${DET_HTTP_PORT}"
  docker run -d \
    --name "${DET_CONTAINER}" \
    --device nvidia.com/gpu=0 \
    -p "${DET_HTTP_PORT}:8080" \
    -p "${DET_MGMT_PORT}:8081" \
    -p "$((DET_MGMT_PORT + 1)):8082" \
    --shm-size=8g \
    --restart unless-stopped \
    "${DET_IMAGE}"

  log "Starting segmentation on GPU 1 → localhost:${SEG_HTTP_PORT}"
  docker run -d \
    --name "${SEG_CONTAINER}" \
    --device nvidia.com/gpu=1 \
    -p "${SEG_HTTP_PORT}:8080" \
    -p "${SEG_MGMT_PORT}:8081" \
    -p "$((SEG_MGMT_PORT + 1)):8082" \
    --shm-size=8g \
    --restart unless-stopped \
    "${SEG_IMAGE}"

  wait_for_healthy "http://127.0.0.1:${DET_HTTP_PORT}/ping" "Detection TorchServe"
  wait_for_healthy "http://127.0.0.1:${SEG_HTTP_PORT}/ping" "Segmentation TorchServe"

  wait_for_model_ready "http://127.0.0.1:${DET_MGMT_PORT}/models/yolov7" "Detection"
  wait_for_model_ready "http://127.0.0.1:${SEG_MGMT_PORT}/models/yolov7" "Segmentation"
}

run_smoke_tests() {
  log "Running smoke tests"

  local args=(
    --det-url "http://127.0.0.1:${DET_HTTP_PORT}"
    --seg-url "http://127.0.0.1:${SEG_HTTP_PORT}"
  )

  if [[ -n "${INFERENCE_IMAGE:-}" ]]; then
    args+=(--det-image "${INFERENCE_IMAGE}")
  fi

  if [[ -n "${SEG_STRIP_IMAGE:-}" ]]; then
    args+=(--seg-strip "${SEG_STRIP_IMAGE}")
  fi

  python3 "${SCRIPT_DIR}/smoke_test_gpu_detectors_updated.py" "${args[@]}"
}

main() {
  (( DO_BUILD )) && build_images
  (( DO_BUILD )) && check_torch_gpu_support "${DET_IMAGE}" "Detection"
  (( DO_BUILD )) && check_torch_gpu_support "${SEG_IMAGE}" "Segmentation"
  (( DO_RUN )) && run_containers
  (( DO_RUN )) && run_smoke_tests

  log "Done."
  log "  Detection:    curl http://127.0.0.1:${DET_HTTP_PORT}/ping"
  log "  Segmentation: curl http://127.0.0.1:${SEG_HTTP_PORT}/ping"
  log "  Optional full inference smoke:"
  log "    INFERENCE_IMAGE=/path/to/shelf.jpg SEG_STRIP_IMAGE=/path/to/strip.jpg ./scripts/deploy_gpu_5090.sh --run-only"
}

main "$@"

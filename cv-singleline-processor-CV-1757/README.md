# cv-singleline-processor

`cv-singleline-processor` is a Python service that consumes image-processing messages from Pub/Sub, runs OCR with Google Vision, extracts SKU data, and publishes normalized SKU payloads to an output Pub/Sub topic.

This service is deployed to **Kubernetes (GKE)**. Legacy Cloud Run deployment instructions are intentionally removed.

## What the service does

1. Subscribes to a Pub/Sub subscription (`SUBSCRIPTION_ID`)
2. Reads image metadata from each message
3. Calls Google Vision OCR on the GCS image URL
4. Extracts SKU and vendor SKU values (including OCR edge-case recovery)
5. Builds SKU + bounding-box payload
6. Publishes results to output topic (`TOPIC_NAME`)
7. Emits Prometheus metrics on `PORT` (default `8082`)

## Runtime and dependencies

- Python runtime: **3.11** (matches Dockerfile base image)
- Main dependencies: Google Cloud Vision, Pub/Sub, Prometheus client
- Container entrypoint: `python3 app.py`

## Required environment variables

The service expects these variables at startup:

- `GCP_PROJECT`
- `TOPIC_NAME`
- `SUBSCRIPTION_ID`
- `EXPERIENCE`
- `SUB_EXPERIENCE`
- `APPLICATION`
- `ENVIRONMENT`
- `PORT` (optional; defaults to `8082`)

## Local development

### 1) Set up Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Run unit tests

```bash
python -m pytest
```

### 3) Run linting

```bash
python -m flake8 .
python -m pylint app.py translators utils metrics.py
```

> Note: lint/test tool versions in this repository are legacy-pinned and may require a compatible Python toolchain to run cleanly.

## Run locally (service mode)

Export required environment variables, then start the service:

```bash
python app.py
```

The service starts a Prometheus endpoint and opens a streaming Pub/Sub subscriber.

## Container build

```bash
docker build -t cv-singleline-processor:local .
```

## CI/CD and deployment

This repo uses GitHub Actions with reusable workflows from `one-thd/cv-tools`:

- `.github/workflows/build-push.yml`
  - Trigger: pull requests to `development`
  - Builds/validates the Python service and pushes container artifacts for the configured registry/project
- `.github/workflows/manual-deploy-environment-variable.yaml`
  - Manual config-map key/value patching for Kubernetes environments

Deployment target is GKE (`sim-np-us-east1-gke-1` by default in manual workflow inputs).

## Observability

Prometheus metrics are defined in `metrics.py` and exported over HTTP on `PORT` (default `8082`).

## Repository layout

- `app.py` — service entry point, Pub/Sub subscriber, fallback routing, metrics, and publishing.
- `new_inference_pipeline_full_image.py` — top-level single-line image pipeline orchestration.
- `pipeline/` — extracted pipeline stages: detection/crops, strips, segmentation, OCR, and OCR assignment.
- `output/` — output payload builders and SKU JSON formatting.
- `legacy/` — legacy full-image OCR parsing and bounding-box helpers.
- `services/` — GCS image I/O, Vertex AI inference wrappers, and metrics definitions.
- `utils/` — shared OCR parsing, validation visualization, and common helpers.
- `translators/` — legacy SKU regex translators.
- `docs/` — project reading guides, method order map, and code segregation plan.
- `test/` — unit tests and fixtures.

## Notes for maintainers

- Cloud Run deployment docs and screenshots were removed because this service is now Kubernetes-based.
- If deployment conventions change again, update this README and `.github/copilot-instructions.md` together.

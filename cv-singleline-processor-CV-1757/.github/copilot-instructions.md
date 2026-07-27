# Copilot Instructions for `cv-singleline-processor`

## Project purpose

This service consumes Pub/Sub image messages, uses Google Vision OCR, extracts SKU data, and republishes normalized SKU events. It runs as a long-lived subscriber process in Kubernetes (GKE), not as Cloud Run.

## Stack and runtime

- Language: Python
- Runtime target: Python 3.11 (Dockerfile base)
- Entrypoint: `python3 app.py`
- Primary integrations:
  - Google Cloud Pub/Sub
  - Google Cloud Vision API
  - Prometheus metrics endpoint

## Core flow

1. `streaming()` creates Pub/Sub streaming subscriber.
2. `process()` handles each message lifecycle and metrics.
3. `process_image()` runs OCR + SKU extraction.
4. `prepare_sku_result_json()` builds output payload.
5. `publish_message()` emits to output topic.

## Critical files

- `/home/runner/work/cv-singleline-processor/cv-singleline-processor/app.py`
- `/home/runner/work/cv-singleline-processor/cv-singleline-processor/translators/atomic/entity_translators.py`
- `/home/runner/work/cv-singleline-processor/cv-singleline-processor/metrics.py`
- `/home/runner/work/cv-singleline-processor/cv-singleline-processor/test/test_app.py`
- `/home/runner/work/cv-singleline-processor/cv-singleline-processor/.github/workflows/build-push.yml`
- `/home/runner/work/cv-singleline-processor/cv-singleline-processor/.github/workflows/manual-deploy-environment-variable.yaml`

## Environment contract

The app expects these env vars to exist at import/runtime:

- `SUBSCRIPTION_ID`
- `GCP_PROJECT`
- `TOPIC_NAME`
- `EXPERIENCE`
- `SUB_EXPERIENCE`
- `APPLICATION`
- `ENVIRONMENT`
- optional: `PORT` (defaults to `8082`)

When adding features, avoid introducing new required env vars unless necessary. If new vars are required, update tests and README in the same change.

## Development rules for agents

1. Keep changes minimal and scoped to the request.
2. Preserve message schema produced by `prepare_sku_result_json()` unless explicitly asked to change it.
3. Do not break SKU extraction edge-case handling in `extract_candidate_skus()` and `detect_bounding_box()`.
4. Keep Pub/Sub ack/nack behavior intact in `streaming()` callback.
5. Do not remove or rename existing metrics without explicit migration guidance.
6. Update README when deployment/runtime behavior changes.
7. Prefer extending existing helpers over duplicating parsing/publishing logic.

## Testing and validation expectations

Before finalizing code changes, run repository checks where available:

```bash
python -m pytest
python -m flake8 .
python -m pylint app.py translators utils metrics.py
```

If environment/toolchain incompatibilities prevent clean execution, document exactly what failed and why.

## Deployment and operations context

- CI for PRs is defined in `build-push.yml` and uses reusable workflows from `one-thd/cv-tools`.
- Manual environment variable updates for Kubernetes ConfigMaps are handled through `manual-deploy-environment-variable.yaml`.
- Keep docs aligned with Kubernetes/GKE deployment assumptions.

## Safe change guidance

- Be careful with global clients instantiated at import time (`vision_client`, `publisher`).
- Maintain compatibility with existing tests that patch Google clients.
- Avoid large refactors unless requested.
- Avoid introducing new dependencies unless required; if added, update `requirements.txt` and mention rationale.

## Documentation hygiene

When docs become stale:

1. Remove obsolete instructions (especially platform migrations, e.g., Cloud Run → Kubernetes).
2. Regenerate README sections to reflect current runtime, CI/CD, and operations.
3. Keep this file synchronized with README and workflow reality.

# Dual TorchServe (Option 1) — one container, two models

One TorchServe process serves:

| Endpoint | MAR | Code tree |
|--|--|--|
| `POST /predictions/detector` | `detector.mar` | `yolov7/` only |
| `POST /predictions/segmenter` | `segmenter.mar` | `yolov7-seg/` only |

Each model runs in its **own worker process** with its own unpacked `model_dir`, so the shared package names `models` / `utils` do not collide.

## How this folder differs

This is the **glue package** that serves both models from **one** TorchServe container. It is not another full YOLO codebase.

| | `yolo7_det_dep_2` | `yolov7-seg` | `torchserve-dual` |
|--|--|--|--|
| Role | Detection source | Segmentation source | Combined deploy |
| Contains `yolov7/` / weights | Yes | Yes | **No** — copies from the other two at Docker build |
| Docker image | `hd-det-gpu` alone | `hd-seg-gpu` alone | **`hd-dual-gpu`** (both) |
| Endpoints | `:9000` only | `:10000` only | Both on **`:9000`** |

## What it contains

- **`Dockerfile`** — builds `detector.mar` + `segmenter.mar`, starts one TorchServe with both
- **`handlers/`** — det-only and seg-only handlers (use each MAR’s `model_dir`, no shared `/app` YOLO path)
- **`packaging/*/common_config_gpu.py`** — per-model config so trees don’t collide
- **`config.properties`** — 1 worker per model
- **`scripts/build_and_run.sh`** — build/run helper
- **`requirements.txt`**, **`README.md`**

At build time it pulls `best.pt` + `yolov7/` from the det folder and `segmentation.pt` + `yolov7-seg/` from the seg folder into **separate** MARs. That is how one GPU server can run both without path confusion.

## GPU checkout path

On `GPU1-A2080`, use the shared data volume (not a laptop path, not only `$HOME`):

```text
/data/<your_user>/HomeDepotCV
```

For `avinash.patel`:

```bash
cd /data/avinash.patel/HomeDepotCV
```

One-time setup (create `/data/$USER`, move or clone) is in `TORCHSERVE_SSH_DEPLOY_172.16.20.100.md` section **0a**.

## Build / run (from repo root on the GPU host)

```bash
cd /data/avinash.patel/HomeDepotCV   # or: cd /data/$USER/HomeDepotCV

# Weights required:
ls cv-singleline-detector-yolo7_det_dep_2/best.pt
ls cv-singleline-detector-yolov7-seg/segmentation.pt

./cv-singleline-torchserve-dual/scripts/build_and_run.sh
```

Or manually:

```bash
cd /data/avinash.patel/HomeDepotCV
docker build -f cv-singleline-torchserve-dual/Dockerfile -t hd-dual-gpu .
docker rm -f hd-dual-gpu hd-det-gpu hd-seg-gpu 2>/dev/null || true
docker run -d --name hd-dual-gpu --gpus all \
  -p 9000:8080 -p 9001:8081 -p 9002:8082 \
  --shm-size=8g --restart unless-stopped hd-dual-gpu
```

## Health

```bash
curl -s http://127.0.0.1:9000/ping
curl -s http://127.0.0.1:9001/models/detector
curl -s http://127.0.0.1:9001/models/segmenter
```

## Smoke test

```bash
cd /data/avinash.patel/HomeDepotCV
python3 scripts/smoke_test_gpu_detectors_updated.py \
  --base-url http://127.0.0.1:9000
```

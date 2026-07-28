# SSH + TorchServe deploy guide — `172.16.20.100`

Use this guide from your **work laptop on corporate VPN** to SSH into the GPU box at `172.16.20.100`, pull the latest HomeDepotCV code (including TorchServe import-collision fixes), rebuild containers, and verify inference.

| Item | Value |
|------|--------|
| Host IP | `172.16.20.100` |
| SSH alias (recommended) | `Ant-PC-2080` |
| SSH user | `avinash.patel` (confirm with your IT admin if login fails) |
| Private key (local only) | `avinash_patel (1).pem` |
| GitHub repo | https://github.com/JediMaster0072/HomeDepotCV |
| Related host (5090 / Streamlit team app) | `172.16.20.108` — different key, see bottom |

**Security:** `*.pem` is in `.gitignore`. Never commit private keys to GitHub.

---

## 0. Prerequisites

On your laptop:

- Corporate VPN connected
- `git`, `ssh`, `docker` (optional locally — Docker runs on the remote host)
- `curl` for health checks
- This repo cloned locally
- PEM file saved outside git or in the repo folder (gitignored)

On the remote host (`172.16.20.100`):

- NVIDIA driver working (`nvidia-smi`)
- Docker with GPU access (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`)
- `gsutil` or pre-downloaded weights (`best.pt`, `segmentation.pt`) if building images
- Enough disk for Docker builds (~10+ GB free recommended)

---

## 1. One-time PEM setup

### macOS / Linux

```bash
# Copy key to a stable location (example)
mkdir -p ~/.ssh
cp "/path/to/avinash_patel (1).pem" ~/.ssh/avinash_patel_100.pem
chmod 400 ~/.ssh/avinash_patel_100.pem
```

Add to `~/.ssh/config`:

```sshconfig
Host Ant-PC-2080
  HostName 172.16.20.100
  User avinash.patel
  IdentityFile ~/.ssh/avinash_patel_100.pem
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
```

### Windows (PowerShell / Git Bash)

```sshconfig
Host Ant-PC-2080
  HostName 172.16.20.100
  User avinash.patel
  IdentityFile C:\Users\<you>\.ssh\avinash_patel_100.pem
  IdentitiesOnly yes
```

Restrict the key file: right-click → Properties → Security → only your user can read.

### Environment variables (used by repo deploy scripts)

```bash
export HOME_DEPOT_SSH_HOST="172.16.20.100"
export HOME_DEPOT_SSH_USER="avinash.patel"
export HOME_DEPOT_SSH_KEY="$HOME/.ssh/avinash_patel_100.pem"
```

---

## 2. Test SSH

```bash
# With SSH config alias
ssh Ant-PC-2080

# Or explicit
ssh -i "$HOME_DEPOT_SSH_KEY" avinash.patel@172.16.20.100
```

Quick remote checks once logged in:

```bash
hostname
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If SSH fails:

1. VPN on?
2. `ping 172.16.20.100`
3. Key permissions `chmod 400`
4. Correct username (`avinash.patel` vs another account)
5. Key authorized on the server

---

## 3. Get latest code on the remote host

### Option A — `git pull` on the server (recommended)

```bash
ssh Ant-PC-2080

cd ~/HomeDepotCV   # adjust if your checkout lives elsewhere
git pull origin main
git log -1 --oneline
```

Clone fresh if needed:

```bash
ssh Ant-PC-2080
git clone https://github.com/JediMaster0072/HomeDepotCV.git
cd HomeDepotCV
```

### Option B — `rsync` from laptop (code only, no PEM)

From your laptop (VPN on):

```bash
LOCAL_REPO=~/Downloads/HOMEDEPOT/HomeDepotCV
REMOTE=Ant-PC-2080
REMOTE_HOME=~/HomeDepotCV

rsync -az --delete \
  --exclude '.git' \
  --exclude '*.pem' \
  --exclude '*.pt' \
  --exclude '*.jpg' \
  --exclude '.venv' \
  -e "ssh -i $HOME_DEPOT_SSH_KEY" \
  "$LOCAL_REPO/" \
  "${REMOTE}:${REMOTE_HOME}/"
```

Do **not** rsync over intern annotation CSVs unless you intend to overwrite team labels.

### TorchServe files changed recently (import collision fix)

These are the files that must be present on the host before rebuild:

```
cv-singleline-detector-yolo7_det_dep_2/common_config_gpu.py
cv-singleline-detector-yolo7_det_dep_2/model_handler.py
cv-singleline-detector-yolo7_det_dep_2/service_pipeline_gpu/stage1_detection.py
cv-singleline-detector-yolo7_det_dep_2/service_pipeline_gpu/stage2_segmentation.py
```

If you use **separate** seg-only deploy tree, mirror the same pattern under `cv-singleline-detector-yolov7-seg/`.

---

## 4. Download model weights (if missing)

On the remote host:

```bash
cd ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
bash model.sh          # downloads best.pt via gsutil

cd ~/HomeDepotCV/cv-singleline-detector-yolov7-seg
bash model.sh          # downloads segmentation.pt
```

Verify:

```bash
ls -lh ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh ~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

---

## 5. Build TorchServe Docker images

SSH into the host, then:

### Detection image

```bash
cd ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
docker build -t hd-det-gpu .
```

The Dockerfile:

1. Installs PyTorch + TorchServe
2. Runs `torch-model-archiver` to create `yolov7.mar`
3. Sets `CMD` to start TorchServe on ports `8080/8081/8082` inside the container

### Segmentation image

```bash
cd ~/HomeDepotCV/cv-singleline-detector-yolov7-seg
docker build -t hd-seg-gpu .
```

### Optional — use the repo deploy helper

From repo root on the host (adapt GPU indices/ports if this box has one GPU):

```bash
cd ~/HomeDepotCV
./scripts/deploy_gpu_5090_updated.sh --build-only
```

> **Note:** `deploy_gpu_5090_updated.sh` was written for the **5090** host (`172.16.20.108`) with two GPUs. On a single-GPU `2080` box, run the manual `docker run` commands below instead of binding `gpu=0` and `gpu=1`.

---

## 6. Start TorchServe containers

### Single-GPU host (typical for `172.16.20.100`)

Stop old containers:

```bash
docker rm -f hd-det-gpu hd-seg-gpu 2>/dev/null || true
```

Start **detection** (host port `9000`):

```bash
docker run -d \
  --name hd-det-gpu \
  --gpus all \
  -p 9000:8080 \
  -p 9001:8081 \
  -p 9002:8082 \
  --shm-size=8g \
  --restart unless-stopped \
  hd-det-gpu
```

Start **segmentation** (host port `10000`):

```bash
docker run -d \
  --name hd-seg-gpu \
  --gpus all \
  -p 10000:8080 \
  -p 10001:8081 \
  -p 10002:8082 \
  --shm-size=8g \
  --restart unless-stopped \
  hd-seg-gpu
```

> If both containers need the same GPU, they will share VRAM. Watch `nvidia-smi` during load.

### What runs inside each container

```bash
torchserve --start --foreground --ncs \
  --ts-config /app/config.properties \
  --model-store model_store \
  --models yolov7=yolov7.mar
```

(Seg image uses `modelstore` — name differs slightly in its Dockerfile.)

---

## 7. Verify health

On the remote host:

```bash
curl -s http://127.0.0.1:9000/ping
curl -s http://127.0.0.1:10000/ping

curl -s http://127.0.0.1:9001/models
curl -s http://127.0.0.1:10001/models
```

Expected ping:

```json
{"status":"Healthy"}
```

Check worker status:

```bash
curl -s http://127.0.0.1:9001/models/yolov7 | python3 -m json.tool
```

Look for `"status": "READY"`.

### Logs

```bash
docker logs -f hd-det-gpu
docker logs -f hd-seg-gpu
```

Successful detection init should show:

```
[Handler] Loading Stage 1 — YOLOv7 detection model …
[Stage1] model loaded | device=cuda …
[Handler] Stage 1 ready.
```

First segmentation request should show:

```
[Handler] Loading Stage 2 — YOLOv7-seg segmentation model …
[Stage2] model loaded …
```

---

## 8. Smoke-test inference

### Detection

From the detection directory (needs `test_image.jpg` or edit path):

```bash
cd ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
# Edit test_torchserve.py URL if needed — default may be localhost:9000
python3 test_torchserve.py
```

Or with `curl` + `request.json`:

```bash
curl -X POST http://127.0.0.1:9000/predictions/yolov7 \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

Request body shape:

```json
{
  "instances": [
    {
      "model_name": "detection",
      "file": "<base64-encoded-png>"
    }
  ]
}
```

### Segmentation

```bash
cd ~/HomeDepotCV
python3 scripts/smoke_test_gpu_detectors_updated.py \
  --det-url http://127.0.0.1:9000 \
  --seg-url http://127.0.0.1:10000 \
  --det-image /path/to/shelf.jpg \
  --seg-strip /path/to/strip.jpg
```

Segmentation request shape:

```json
{
  "instances": [
    {
      "model_name": "segmentation",
      "strip_id": 0,
      "file": "<base64-encoded-png>"
    }
  ]
}
```

---

## 9. Full update workflow (copy/paste checklist)

Run from your **work laptop** unless noted.

```bash
# 1) VPN on
# 2) SSH works
ssh Ant-PC-2080 'hostname && nvidia-smi -L'

# 3) Pull latest code ON THE SERVER
ssh Ant-PC-2080 'cd ~/HomeDepotCV && git pull origin main'

# 4) Rebuild + restart ON THE SERVER
ssh Ant-PC-2080 'bash -s' <<'REMOTE'
set -euo pipefail
cd ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
[[ -f best.pt ]] || bash model.sh
docker build -t hd-det-gpu .
cd ~/HomeDepotCV/cv-singleline-detector-yolov7-seg
[[ -f segmentation.pt ]] || bash model.sh
docker build -t hd-seg-gpu .
docker rm -f hd-det-gpu hd-seg-gpu 2>/dev/null || true
docker run -d --name hd-det-gpu --gpus all \
  -p 9000:8080 -p 9001:8081 -p 9002:8082 \
  --shm-size=8g --restart unless-stopped hd-det-gpu
docker run -d --name hd-seg-gpu --gpus all \
  -p 10000:8080 -p 10001:8081 -p 10002:8082 \
  --shm-size=8g --restart unless-stopped hd-seg-gpu
sleep 15
curl -sf http://127.0.0.1:9000/ping
curl -sf http://127.0.0.1:10000/ping
REMOTE

# 5) Optional smoke test ON THE SERVER
ssh Ant-PC-2080 'cd ~/HomeDepotCV && python3 scripts/smoke_test_gpu_detectors_updated.py --det-url http://127.0.0.1:9000 --seg-url http://127.0.0.1:10000'
```

---

## 10. Port reference

| Service | Container | Inference | Management | Metrics |
|---------|-----------|-----------|------------|---------|
| Detection | `hd-det-gpu` | `9000` → `8080` | `9001` → `8081` | `9002` → `8082` |
| Segmentation | `hd-seg-gpu` | `10000` → `8080` | `10001` → `8081` | `10002` → `8082` |

From another machine on VPN (replace with server hostname if DNS exists):

```bash
curl http://172.16.20.100:9000/ping
curl http://172.16.20.100:10000/ping
```

Firewall rules on the host must allow these ports if you call from outside localhost.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Permission denied (publickey)` | Wrong PEM or user | Check `IdentityFile`, `chmod 400`, username |
| `ping` fails | VPN off | Connect corporate VPN |
| Docker has no GPU | NVIDIA runtime / CDI | See `gpu-docker-cdi-fix.md` (written for `.108`, same class of issue) |
| `sm_120 NOT supported` | Wrong PyTorch image for GPU gen | 5090 needs CUDA 12.8+ PyTorch; 2080 may need different base |
| Segmentation imports detection `utils` | Module cache collision | Ensure latest `common_config_gpu.py` + stage files are deployed |
| Container exits immediately | Missing weights / bad `.mar` | `docker logs hd-det-gpu` |
| Port already in use | Old container running | `docker rm -f hd-det-gpu hd-seg-gpu` |

### Confirm import-collision fix is live

On the server:

```bash
grep -n "purge_yolo_modules\|activate_det_repo\|activate_seg_repo" \
  ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/common_config_gpu.py
```

You should see those function names. Rebuild Docker images after pulling — a running container does not pick up file changes until rebuilt.

---

## 12. Do not commit the PEM

```bash
# This should show the pem as ignored:
cd ~/HomeDepotCV
git check-ignore -v "avinash_patel (1).pem"
```

Expected: matched by `*.pem` in `.gitignore`.

Store the key in `~/.ssh/` and reference it only via `HOME_DEPOT_SSH_KEY` or SSH config.

---

## 13. Other GPU host (`172.16.20.108` — 5090 / Streamlit)

The **team annotation app** and original 5090 TorchServe docs target a different machine:

| | `172.16.20.100` | `172.16.20.108` |
|--|-----------------|-----------------|
| Role | New / 2080 TorchServe testing | 5090 production + Streamlit `:8503` |
| PEM (Avinash) | `avinash_patel (1).pem` | `avinash_patel_lf.pem` |
| Streamlit team URL | — | http://172.16.20.108:8503 |

Deploy scripts under `cv-singleline-processor-CV-1757/scripts/deploy/` default to `.108`. Override host/key:

```bash
export HOME_DEPOT_SSH_HOST="172.16.20.100"
export HOME_DEPOT_SSH_KEY="$HOME/.ssh/avinash_patel_100.pem"
```

---

## 14. Quick reference

```bash
# Connect
ssh Ant-PC-2080

# Pull + rebuild TorchServe (on server)
cd ~/HomeDepotCV && git pull origin main
cd cv-singleline-detector-yolo7_det_dep_2 && docker build -t hd-det-gpu .
cd ../cv-singleline-detector-yolov7-seg && docker build -t hd-seg-gpu .

# Restart
docker rm -f hd-det-gpu hd-seg-gpu
docker run -d --name hd-det-gpu --gpus all -p 9000:8080 -p 9001:8081 --shm-size=8g hd-det-gpu
docker run -d --name hd-seg-gpu --gpus all -p 10000:8080 -p 10001:8081 --shm-size=8g hd-seg-gpu

# Health
curl http://127.0.0.1:9000/ping
curl http://127.0.0.1:10000/ping
```

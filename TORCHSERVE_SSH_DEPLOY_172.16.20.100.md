# SSH + TorchServe deploy guide — `172.16.20.100`

Use this guide from your **work laptop on corporate VPN** to SSH into the GPU box at `172.16.20.100`, pull the latest HomeDepotCV code (including TorchServe import-collision fixes), rebuild containers, and verify inference.

| Item | Value |
|------|--------|
| Host IP | `172.16.20.100` |
| SSH alias (recommended) | `Ant-PC-2080` |
| Remote hostname | `GPU1-A2080` |
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

## 3. Get latest code on the remote host (`GPU1-A2080`)

If you see:

```text
cd: /home/avinash.patel/HomeDepotCV: No such file or directory
fatal: not a git repository (or any of the parent directories): .git
```

the repo has **not been cloned yet** on that machine. Run the **first-time setup** below once, then use `git pull` on future updates.

---

### 3a. First-time git setup on `GPU1-A2080` (run on the server)

SSH in:

```bash
ssh Ant-PC-2080
# you should see a prompt like: avinash.patel@GPU1-A2080:~$
```

#### Step 1 — Install git (if missing)

```bash
git --version
```

If that fails:

```bash
sudo apt-get update
sudo apt-get install -y git
```

#### Step 2 — Clone the GitHub repo (HTTPS — no GitHub login required for pull)

The repo is public. Clone it into your home directory:

```bash
cd ~
git clone https://github.com/JediMaster0072/HomeDepotCV.git
cd ~/HomeDepotCV
```

Expected:

```text
Cloning into 'HomeDepotCV'...
```

#### Step 3 — Verify the remote and branch

```bash
cd ~/HomeDepotCV
git remote -v
git branch
git log -1 --oneline
```

You should see:

```text
origin  https://github.com/JediMaster0072/HomeDepotCV.git (fetch)
origin  https://github.com/JediMaster0072/HomeDepotCV.git (push)
* main
158c25e ...   # commit hash will vary
```

#### Step 4 — Optional git identity (only needed if you will commit from this host)

```bash
git config --global user.name "Avinash Patel"
git config --global user.email "your.email@example.com"
```

Not required for `git pull` / `git clone` only.

#### Step 5 — Confirm TorchServe directories exist

```bash
ls ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
ls ~/HomeDepotCV/cv-singleline-detector-yolov7-seg
ls ~/HomeDepotCV/scripts/deploy_gpu_5090_updated.sh
```

If those paths exist, you are ready for **Section 4** (weights) and **Section 5** (Docker build).

---

### 3b. Update an existing clone (`git pull`)

Only run this **after** `~/HomeDepotCV` already exists from section 3a:

```bash
ssh Ant-PC-2080

cd ~/HomeDepotCV
git pull origin main
git log -1 --oneline
```

If `git pull` asks for credentials on HTTPS, the GPU host may not have outbound GitHub access. Use **Option C (rsync)** below instead.

---

### Option C — `rsync` from laptop (no git on server required)

From your laptop (VPN on) if the GPU cannot reach GitHub:

```bash
LOCAL_REPO=~/Downloads/HOMEDEPOT/HomeDepotCV
REMOTE=Ant-PC-2080
REMOTE_HOME=~/HomeDepotCV

ssh Ant-PC-2080 "mkdir -p ~/HomeDepotCV"

rsync -az \
  --exclude '.git' \
  --exclude '*.pem' \
  --exclude '*.pt' \
  --exclude '*.jpg' \
  --exclude '.venv' \
  -e "ssh" \
  "$LOCAL_REPO/" \
  "${REMOTE}:${REMOTE_HOME}/"
```

Do **not** rsync over intern annotation CSVs unless you intend to overwrite team labels.

---

### Option D — GitHub SSH from the GPU (optional, for push access)

Only needed if you want to **push commits from** `GPU1-A2080`, not for `git pull`.

On the GPU host:

```bash
ssh-keygen -t ed25519 -C "avinash.patel@GPU1-A2080" -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub
```

Add the printed public key in GitHub → **Settings → SSH and GPG keys → New SSH key**.

Then on the GPU:

```bash
cd ~
git clone git@github.com:JediMaster0072/HomeDepotCV.git
# or switch an existing HTTPS clone:
cd ~/HomeDepotCV
git remote set-url origin git@github.com:JediMaster0072/HomeDepotCV.git
git pull origin main
```

---

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

`model.sh` uses `gsutil` to pull weights from Google Cloud Storage. If you see:

```text
model.sh: line 4: gsutil: command not found
```

use **Option A** (install `gsutil`) or **Option B** (copy from your Mac).

---

### Option A — Install `gsutil` on `GPU1-A2080` (recommended)

On the GPU host:

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update
sudo apt-get install -y google-cloud-cli
gsutil version
```

Then download weights:

```bash
cd ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
bash model.sh

cd ~/HomeDepotCV/cv-singleline-detector-yolov7-seg
bash model.sh
```

If `gsutil cp` fails with **AccessDenied** or **401**, the bucket is private. Use **Option B** or ask your team for `gcloud auth login` / a service-account key with read access to `gs://selling-pipeline-ml-models/`.

Manual `gsutil` paths (same as `model.sh`):

```bash
# Detection (~136 MB)
gsutil cp gs://selling-pipeline-ml-models/singleline-pipeline-det-models-1.2/best.pt \
  ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

# Segmentation (~454 MB)
gsutil cp gs://selling-pipeline-ml-models/singleline-pipeline-seg-models-1.1/segmentation.pt \
  ~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

---

### Option B — Copy weights from your Mac (`scp`)

**Run these commands on your Mac (laptop), not on `GPU1-A2080`.**  
`scp` pushes files **from the Mac → GPU**. If you run `scp` while already SSH'd into the GPU, `$WORKDIR` will be empty and you will get `No such file or directory`.

If you still have the original zip (`HomeDepotCV 2.zip`) on your laptop, extract only the `.pt` files:

**On your Mac** (VPN on, new Terminal window — do **not** be SSH'd into the GPU):

```bash
ZIP="/Users/avinash_a_patel/Downloads/HOMEDEPOT/HomeDepotCV 2.zip"
WORKDIR=/tmp/hd-weights
mkdir -p "$WORKDIR"

unzip -j "$ZIP" "HomeDepotCV/singleline-pipeline-det-models-1.2_best.pt" -d "$WORKDIR"
unzip -j "$ZIP" "HomeDepotCV/singleline-pipeline-seg-models_segmentation.pt" -d "$WORKDIR"
mv "$WORKDIR/singleline-pipeline-det-models-1.2_best.pt" "$WORKDIR/best.pt"
mv "$WORKDIR/singleline-pipeline-seg-models_segmentation.pt" "$WORKDIR/segmentation.pt"

ls -lh "$WORKDIR"/*.pt
```

Copy to the GPU (still on your Mac):

```bash
scp "$WORKDIR/best.pt" \
  avinash.patel@172.16.20.100:~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

scp "$WORKDIR/segmentation.pt" \
  avinash.patel@172.16.20.100:~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

If you use SSH config alias `Ant-PC-2080`:

```bash
scp "$WORKDIR/best.pt" \
  Ant-PC-2080:~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

scp "$WORKDIR/segmentation.pt" \
  Ant-PC-2080:~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

**Then** SSH back into the GPU and verify:

```bash
ssh Ant-PC-2080
ls -lh ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh ~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

If you already have `.pt` files elsewhere on the Mac, `scp` them directly to those two remote paths.

---

### Verify on the GPU

```bash
ls -lh ~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh ~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

Rough expected sizes:

| File | ~Size |
|------|--------|
| `best.pt` | 136 MB |
| `segmentation.pt` | 454 MB |

If either file is missing or tiny, do not run `docker build` yet.

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

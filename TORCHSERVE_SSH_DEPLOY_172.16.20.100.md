# SSH + TorchServe deploy guide — `172.16.20.100`

Use this guide from your **work laptop on corporate VPN** to SSH into the GPU box at `172.16.20.100`, pull the latest HomeDepotCV code, and run **one TorchServe container** that serves both detection and segmentation (Option 1: two `.mar` files, separate worker processes).

| Item | Value |
|------|--------|
| Host IP | `172.16.20.100` |
| SSH alias (recommended) | `Ant-PC-2080` |
| Remote hostname | `GPU1-A2080` |
| SSH user | `avinash.patel` (confirm with your IT admin if login fails) |
| Private key (local only) | `avinash_patel (1).pem` |
| GitHub repo | https://github.com/JediMaster0072/HomeDepotCV |
| Preferred deploy | `hd-dual-gpu` — one container, ports `9000/9001/9002` |
| Endpoints | `POST /predictions/detector` and `POST /predictions/segmenter` |
| Related host (5090 / Streamlit team app) | `172.16.20.108` — different key, see bottom |
| Repo on GPU (preferred) | `/data/avinash.patel/HomeDepotCV` |

**Security:** `*.pem` is in `.gitignore`. Never commit private keys to GitHub.


**Do not use** a Mac path like `/Users/.../Downloads/HomeDepotCV` on the GPU. Teammates cannot see your laptop disk. On `GPU1-A2080`, keep the checkout under `/data/<your_user>/HomeDepotCV` so it lives on the shared data volume.

---

## 0a. One-time: put the repo under `/data` (on the GPU)

SSH in, then:

```bash
# Create your data directory (ask admin if /data is root-owned)
sudo mkdir -p /data/$USER
sudo chown -R $USER:$USER /data/$USER

# Option A — move existing home checkout (keeps weights + containers build context)
if [ -d "$HOME/HomeDepotCV" ] && [ ! -d "/data/avinash.patel/HomeDepotCV" ]; then
  mv "$HOME/HomeDepotCV" "/data/avinash.patel/HomeDepotCV"
fi

# Option B — fresh clone
mkdir -p /data/$USER
cd /data/$USER
git clone https://github.com/JediMaster0072/HomeDepotCV.git
# If you already moved/cloned as avinash.patel, use:
# cd /data/avinash.patel/HomeDepotCV

cd /data/avinash.patel/HomeDepotCV
git pull origin main

# Confirm weights still present after a move
ls -lh cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh cv-singleline-detector-yolov7-seg/segmentation.pt
```

From then on, **always**:

```bash
cd /data/avinash.patel/HomeDepotCV
```

Optional convenience symlink from home:

```bash
ln -sfn /data/avinash.patel/HomeDepotCV ~/HomeDepotCV
```


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
avinash.patel@Avinash-Patel HomeDepotCV % mkdir -p ~/.ssh
cp "/Users/avinash.patel/Downloads/HomeDepotCV/avinash_patel (1).pem" ~/.ssh/avinash_patel_100.pem
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
cd: /data/avinash.patel/HomeDepotCV: No such file or directory
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
cd /data/avinash.patel/HomeDepotCV
```

Expected:

```text
Cloning into 'HomeDepotCV'...
```

#### Step 3 — Verify the remote and branch

```bash
cd /data/avinash.patel/HomeDepotCV
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
ls /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
ls /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg
ls /data/avinash.patel/HomeDepotCV/scripts/deploy_gpu_5090_updated.sh
```

If those paths exist, you are ready for **Section 4** (weights) and **Section 5** (Docker build).

---

### 3b. Update an existing clone (`git pull`)

Only run this **after** `/data/avinash.patel/HomeDepotCV` already exists from section 3a:

```bash
ssh Ant-PC-2080

cd /data/avinash.patel/HomeDepotCV
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
REMOTE_HOME=/data/avinash.patel/HomeDepotCV

ssh Ant-PC-2080 "mkdir -p /data/avinash.patel/HomeDepotCV"

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
cd /data/avinash.patel/HomeDepotCV
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

Weights (`best.pt`, `segmentation.pt`) are **not** in GitHub — they are large binaries and are gitignored. `git pull` alone will never create them.

`model.sh` tries to download from Google Cloud Storage via `gsutil`. Common failures:

| Error | Meaning | What to do |
|-------|---------|------------|
| `gsutil: command not found` | Cloud SDK not installed | Option A |
| `401 Anonymous caller` / `Permission denied` | Bucket is private; GPU not logged into GCP | Option C (preferred) or `gcloud auth login` |
| `unzip: cannot find or open ... HomeDepotCV 2.zip` | Original zip is gone / wrong path on this Mac | Option C, or find/re-obtain the zip (Option B) |
| `scp: No such file or directory` while on GPU prompt | `scp` was run **on the GPU** | Run `scp` from a **Mac** terminal instead |

---

### Where to run `scp`

You can run `scp` from **any folder on your Mac** (`cd ~` is fine). The working directory does not matter if you use absolute paths.

**Must be a local Mac terminal** — prompt like `avinash.patel@Avinash-Patel ~ %`  
**Not** `avinash.patel@GPU1-A2080:...$`

---

### Option A — Install `gsutil` on `GPU1-A2080`

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
cd /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
bash model.sh

cd /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg
bash model.sh
```

If `gsutil cp` fails with **401 Anonymous caller**, the bucket is private. Authenticate (if you have HD GCP access):

```bash
gcloud auth login
# then re-run bash model.sh
```

Or skip GCS and use **Option C**.

Manual `gsutil` paths (same as `model.sh`):

```bash
# Detection (~136 MB)
gsutil cp gs://selling-pipeline-ml-models/singleline-pipeline-det-models-1.2/best.pt \
  /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

# Segmentation (~454 MB)
gsutil cp gs://selling-pipeline-ml-models/singleline-pipeline-seg-models-1.1/segmentation.pt \
  /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

---

### Option B — Copy weights from your Mac (`scp`)

**Only unzip if you do not already have `.pt` files on the Mac.**

If `find` already shows weights (common after cloning/extracting `HomeDepotCV` locally), **skip unzip** and `scp` them directly:

```bash
# On your Mac — check what you already have
find ~/Downloads -name 'best.pt' -o -name 'segmentation.pt' 2>/dev/null | head
```

Example when files exist under `~/Downloads/HomeDepotCV/`:

```bash
scp ~/Downloads/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt \
  Ant-PC-2080:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

scp ~/Downloads/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt \
  Ant-PC-2080:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

Verify on the GPU:

```bash
ssh Ant-PC-2080
ls -lh /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

---

#### Option B2 — Extract from zip (only if `.pt` files are not already on the Mac)

**Run on your Mac (laptop), not on `GPU1-A2080`.**

First locate the zip:

```bash
find ~/Downloads -maxdepth 4 -iname '*HomeDepotCV*.zip' 2>/dev/null
```

Use the **real path** from `find` — do not leave the placeholder `/path/from/find/...` in the command.

Example for `avinash.patel` home directory:

```bash
cd ~
ZIP="$HOME/Downloads/HomeDepotCV 2.zip"   # real path from find
WORKDIR=/tmp/hd-weights
mkdir -p "$WORKDIR"

unzip -j "$ZIP" "HomeDepotCV/singleline-pipeline-det-models-1.2_best.pt" -d "$WORKDIR"
unzip -j "$ZIP" "HomeDepotCV/singleline-pipeline-seg-models_segmentation.pt" -d "$WORKDIR"
mv "$WORKDIR/singleline-pipeline-det-models-1.2_best.pt" "$WORKDIR/best.pt"
mv "$WORKDIR/singleline-pipeline-seg-models_segmentation.pt" "$WORKDIR/segmentation.pt"
ls -lh "$WORKDIR"/*.pt
```

Copy to the 2080:

```bash
scp /tmp/hd-weights/best.pt \
  Ant-PC-2080:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

scp /tmp/hd-weights/segmentation.pt \
  Ant-PC-2080:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

**Common mistake:** pasting the doc placeholder literally:

```bash
ZIP="/path/from/find/HomeDepotCV 2.zip"   # WRONG — replace with real path
```

That will always fail with `cannot find or open`.

---

### Option C — Copy weights from the 5090 host (`172.16.20.108`) → 2080 (`172.16.20.100`)

**Preferred when the Mac zip is gone and GCS is private.**  
If TorchServe was already deployed on `GPU5-A5090` (`172.16.20.108`), the `.pt` files are often already there.

**First, check that `.108` has the weights** (from your Mac):

```bash
ssh -i ~/.ssh/avinash_patel_lf.pem avinash.patel@172.16.20.108 \
  'ls -lh /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt \
         /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt 2>&1'
```

(Adjust key/user if your `.108` access differs.)

**Then copy Mac ← 5090 ← then → 2080** (run all of this on your Mac):

```bash
# Detection
scp -i ~/.ssh/avinash_patel_lf.pem \
  avinash.patel@172.16.20.108:~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt \
  /tmp/best.pt

scp /tmp/best.pt \
  Ant-PC-2080:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

# Segmentation
scp -i ~/.ssh/avinash_patel_lf.pem \
  avinash.patel@172.16.20.108:~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt \
  /tmp/segmentation.pt

scp /tmp/segmentation.pt \
  Ant-PC-2080:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

Alternative: one-hop via `scp -3` (Mac as relay, files never need a local zip):

```bash
ssh Ant-PC-2080 \
  'mkdir -p /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2 \
            /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg'

scp -3 -i ~/.ssh/avinash_patel_lf.pem \
  avinash.patel@172.16.20.108:~/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt \
  avinash.patel@172.16.20.100:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt

scp -3 -i ~/.ssh/avinash_patel_lf.pem \
  avinash.patel@172.16.20.108:~/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt \
  avinash.patel@172.16.20.100:/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

If `.108` also does not have the weights, re-obtain `HomeDepotCV 2.zip` from whoever shared it, or get GCP bucket access and use Option A with `gcloud auth login`.

---

### Verify on the GPU

```bash
ssh Ant-PC-2080
ls -lh /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

Rough expected sizes:

| File | ~Size |
|------|--------|
| `best.pt` | 136 MB |
| `segmentation.pt` | 454 MB |

Destination paths on `GPU1-A2080`:

| File | Remote path |
|------|-------------|
| `best.pt` | `/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt` |
| `segmentation.pt` | `/data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt` |

If either file is missing or tiny, do not run `docker build` yet.

---

## 5. Build TorchServe Docker image (preferred: dual Option 1)

SSH into the host, then:

> If you get `permission denied ... docker.sock`, prefix with `sudo` or add your user to the `docker` group (`sudo usermod -aG docker $USER` then re-login / `newgrp docker`).

### Dual image — one TorchServe, two MARs (`hd-dual-gpu`)

This is the **preferred** path: one container on this single GPU server serves both models. Each model is a separate `.mar` with its own worker process and `model_dir` (`yolov7/` never shares a process with `yolov7-seg/`).

Confirm weights:

```bash
ls -lh /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg/segmentation.pt
```

Build from **repo root** (Dockerfile copies both trees into separate MAR staging dirs):

```bash
cd /data/avinash.patel/HomeDepotCV
git pull origin main
chmod +x cv-singleline-torchserve-dual/scripts/build_and_run.sh
./cv-singleline-torchserve-dual/scripts/build_and_run.sh --build-only
# or:
# docker build -f cv-singleline-torchserve-dual/Dockerfile -t hd-dual-gpu .
```

What the dual Dockerfile does:

1. Installs CUDA PyTorch + TorchServe
2. Builds `detector.mar` (best.pt + yolov7/ + detection handler)
3. Builds `segmenter.mar` (segmentation.pt + yolov7-seg/ + segmentation handler)
4. Starts **one** TorchServe with `--models detector=detector.mar,segmenter=segmenter.mar`

See also `cv-singleline-torchserve-dual/README.md`.

### Fallback — separate det/seg images (legacy)

```bash
cd /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2 && sudo docker build -t hd-det-gpu .
cd /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg && sudo docker build -t hd-seg-gpu .
```

---

## 6. Start TorchServe (preferred: one dual container)

### Dual container on `172.16.20.100`

Stop old split **and** dual containers:

```bash
docker rm -f hd-dual-gpu hd-det-gpu hd-seg-gpu 2>/dev/null || true
```

Start **one** dual service (host ports `9000/9001/9002`):

```bash
cd /data/avinash.patel/HomeDepotCV
./cv-singleline-torchserve-dual/scripts/build_and_run.sh
```

Or manually:

```bash
docker run -d \
  --name hd-dual-gpu \
  --gpus all \
  -p 9000:8080 \
  -p 9001:8081 \
  -p 9002:8082 \
  --shm-size=8g \
  --restart unless-stopped \
  hd-dual-gpu
```

Both models load onto the same GPU (one worker each). Watch `nvidia-smi` during load.

Inside the dual container:

```bash
torchserve --start --foreground --ncs \
  --ts-config /app/config.properties \
  --model-store /app/model_store \
  --models detector=detector.mar,segmenter=segmenter.mar
```

**Do not run `torchserve` on the host.** It runs inside Docker.

### Fallback — two containers (legacy)

```bash
docker rm -f hd-det-gpu hd-seg-gpu 2>/dev/null || true
docker run -d --name hd-det-gpu --gpus all \
  -p 9000:8080 -p 9001:8081 -p 9002:8082 \
  --shm-size=8g --restart unless-stopped hd-det-gpu
docker run -d --name hd-seg-gpu --gpus all \
  -p 10000:8080 -p 10001:8081 -p 10002:8082 \
  --shm-size=8g --restart unless-stopped hd-seg-gpu
```

---

## 7. Verify health

### Confirm the dual container is running

```bash
docker ps --filter name=hd-dual
docker logs --tail 120 hd-dual-gpu
```

Look for both workers initializing (detection Stage 1 and segmentation Stage 2).

### Health checks (on the GPU) — dual

```bash
curl -s http://127.0.0.1:9000/ping
curl -s http://127.0.0.1:9001/models
curl -s http://127.0.0.1:9001/models/detector | python3 -m json.tool
curl -s http://127.0.0.1:9001/models/segmenter | python3 -m json.tool
```

Expect `"status":"Healthy"` and both `detector` and `segmenter` with workers `"status": "READY"`. First load can take a few minutes.

Successful dual logs should include lines like:

```
[DetectionHandler] Loading Stage 1 — YOLOv7 detection …
[DetectionHandler] Stage 1 ready …
[SegmentationHandler] Loading Stage 2 — YOLOv7-seg …
[SegmentationHandler] Stage 2 ready …
```

### Legacy split-container checks

```bash
curl -s http://127.0.0.1:9000/ping
curl -s http://127.0.0.1:10000/ping
curl -s http://127.0.0.1:9001/models/yolov7 | python3 -m json.tool
curl -s http://127.0.0.1:10001/models/yolov7 | python3 -m json.tool
```

---

## 8. Smoke-test inference

### Dual (preferred)

```bash
cd /data/avinash.patel/HomeDepotCV
python3 scripts/smoke_test_gpu_detectors_updated.py \
  --base-url http://127.0.0.1:9000
```

With optional images:

```bash
python3 scripts/smoke_test_gpu_detectors_updated.py \
  --base-url http://127.0.0.1:9000 \
  --det-image /path/to/shelf.jpg \
  --seg-strip /path/to/strip.png
```

Manual curls:

```bash
# Detection
curl -X POST http://127.0.0.1:9000/predictions/detector \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"model_name":"detection","file":"<base64>"}]}'

# Segmentation
curl -X POST http://127.0.0.1:9000/predictions/segmenter \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"model_name":"segmentation","strip_id":0,"file":"<base64>"}]}'
```

### Legacy two-URL mode

```bash
python3 scripts/smoke_test_gpu_detectors_updated.py \
  --det-url http://127.0.0.1:9000 \
  --seg-url http://127.0.0.1:10000
```

---

## 9. Full update workflow (copy/paste checklist)

Run from your **work laptop** unless noted.

```bash
# 1) VPN on
# 2) SSH works
ssh Ant-PC-2080 'hostname && nvidia-smi -L'

# 3) Pull latest code ON THE SERVER
ssh Ant-PC-2080 'cd /data/avinash.patel/HomeDepotCV && git pull origin main'

# 4) Rebuild + restart dual ON THE SERVER
ssh Ant-PC-2080 'bash -s' <<'REMOTE'
set -euo pipefail
cd /data/avinash.patel/HomeDepotCV
ls -lh cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh cv-singleline-detector-yolov7-seg/segmentation.pt
chmod +x cv-singleline-torchserve-dual/scripts/build_and_run.sh
./cv-singleline-torchserve-dual/scripts/build_and_run.sh
REMOTE

# 5) Smoke test ON THE SERVER
ssh Ant-PC-2080 'cd /data/avinash.patel/HomeDepotCV && python3 scripts/smoke_test_gpu_detectors_updated.py --base-url http://127.0.0.1:9000'
```

---

## 10. Port reference

| Service | Container | Inference | Management | Metrics |
|---------|-----------|-----------|------------|---------|
| **Dual (preferred)** | `hd-dual-gpu` | `9000` → `8080` | `9001` → `8081` | `9002` → `8082` |
| Detection (legacy) | `hd-det-gpu` | `9000` → `8080` | `9001` → `8081` | `9002` → `8082` |
| Segmentation (legacy) | `hd-seg-gpu` | `10000` → `8080` | `10001` → `8081` | `10002` → `8082` |

Dual model URLs (same host port):

- `http://172.16.20.100:9000/predictions/detector`
- `http://172.16.20.100:9000/predictions/segmenter`

```bash
curl http://172.16.20.100:9000/ping
```

Firewall rules on the host must allow these ports if you call from outside localhost.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Permission denied (publickey)` | Wrong PEM or user | Check `IdentityFile`, `chmod 400`, username |
| `ping` fails | VPN off | Connect corporate VPN |
| Docker has no GPU | NVIDIA runtime / CDI | See `gpu-docker-cdi-fix.md` |
| Seg init: `Weights only load failed` / `weights_only` | PyTorch 2.6+ default | `yolov7-seg/models/experimental.py` must use `torch.load(..., weights_only=False)` then rebuild |
| Container exits immediately | Missing weights / bad `.mar` | `docker logs hd-dual-gpu` |
| Port already in use | Old container running | `docker rm -f hd-dual-gpu hd-det-gpu hd-seg-gpu` |
| Dual OOM on 2080 | Both models in VRAM | Keep 1 worker/model; reduce concurrency; check `nvidia-smi` |
| Import confusion / wrong `models`/`utils` | Both YOLO trees on shared PYTHONPATH | Dual image must not set `PYTHONPATH=/app` with both repos; use MAR `model_dir` only |

### Confirm dual package is present

```bash
ls /data/avinash.patel/HomeDepotCV/cv-singleline-torchserve-dual/Dockerfile
ls /data/avinash.patel/HomeDepotCV/cv-singleline-torchserve-dual/handlers/
```

Rebuild after `git pull` — a running container does not pick up file changes until rebuilt.

---

## 12. Do not commit the PEM

```bash
# This should show the pem as ignored:
cd /data/avinash.patel/HomeDepotCV
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



## 14. Quick reference (dual)

```bash
# Connect
ssh Ant-PC-2080

# Pull + rebuild dual TorchServe (on server)
cd /data/avinash.patel/HomeDepotCV && git pull origin main
./cv-singleline-torchserve-dual/scripts/build_and_run.sh

# Health
curl http://127.0.0.1:9000/ping
curl http://127.0.0.1:9001/models/detector
curl http://127.0.0.1:9001/models/segmenter

# Smoke
python3 scripts/smoke_test_gpu_detectors_updated.py --base-url http://127.0.0.1:9000
```

## 15. Longer-term note (Triton)

TorchServe Option 1 meets the single-GPU-server goal today. TorchServe is in limited upstream maintenance; if you later need a longer-lived multi-model server on the same host, evaluate **NVIDIA Triton** (model repository + separate backends per model) without splitting across two GPU servers. LitServe is a lighter Python option. No migration is required for the current dual MAR deploy.

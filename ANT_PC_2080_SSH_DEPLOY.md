# Ant-PC-2080 SSH & Deploy Guide

**Host:** `Ant-PC-2080` (`172.16.20.100`)  
**User:** `vaibhav.singh`  
**Key (local copy):** `vaibhav_singh_100.pem` (gitignored — do not commit)

## Can this be done from Cursor on this Mac?

**No — not from the current environment.**  
`172.16.20.100` is on the corporate internal network. From this machine the host does not respond (VPN / corp network required). Use your **work laptop on VPN** (or any machine that can reach `172.16.20.*`).

| Machine | IP | Notes |
|--------|-----|--------|
| Ant-PC-2080 | `172.16.20.100` | This guide (Vaibhav key) |
| GPU5-A5090 | `172.16.20.108` | Existing 5090 TorchServe / Streamlit (`:8503`) |

---

## 1. One-time SSH setup (work laptop)

### Windows (`~/.ssh/config`)

```sshconfig
Host Ant-PC-2080
  HostName 172.16.20.100
  User vaibhav.singh
  IdentityFile C:\Users\vaibhav.singh\.ssh\vaibhav_singh_100.pem
```

Put the `.pem` at that path (or change `IdentityFile` to wherever you keep it).

### macOS

1. Copy the key into the repo (or `~/.ssh/`), then lock permissions:

```bash
chmod 400 /path/to/vaibhav_singh_100.pem
```

2. Add to `~/.ssh/config`:

```sshconfig
Host Ant-PC-2080
  HostName 172.16.20.100
  User vaibhav.singh
  IdentityFile /path/to/vaibhav_singh_100.pem
```

### Connect

```bash
# After VPN is up:
ssh Ant-PC-2080

# Or without config:
ssh -i /path/to/vaibhav_singh_100.pem vaibhav.singh@172.16.20.100
```

If SSH fails:

1. Confirm VPN is connected.
2. `ping 172.16.20.100`
3. Check key permissions (`chmod 400` on Mac/Linux).
4. Confirm your account is authorized for this host (key must match `vaibhav.singh`).

---

## 2. What you typically deploy from the laptop

### A) SKU annotation Streamlit fix (digits + `X` placeholders)

Code lives under:

- `cv-singleline-processor-CV-1757/scripts/common/sku_review.py`
- `cv-singleline-processor-CV-1757/scripts/golden_dataset/streamlit_expected_sku_review.py`

**Do not overwrite remote annotation CSVs** when syncing code. Prefer `rsync` of script paths only, or `git pull` on the host if the remote checkout tracks the same repo.

Example (from your laptop, VPN on), sync **code only**:

```bash
# Adjust LOCAL_REPO and REMOTE_HOME to match your layout
LOCAL_REPO=~/Downloads/HOMEDEPOT/HomeDepotCV
REMOTE=Ant-PC-2080
REMOTE_HOME=~/HomeDepotCV   # change if the checkout path differs

rsync -az -e ssh \
  "$LOCAL_REPO/cv-singleline-processor-CV-1757/scripts/common/sku_review.py" \
  "$REMOTE:$REMOTE_HOME/cv-singleline-processor-CV-1757/scripts/common/sku_review.py"

rsync -az -e ssh \
  "$LOCAL_REPO/cv-singleline-processor-CV-1757/scripts/golden_dataset/streamlit_expected_sku_review.py" \
  "$REMOTE:$REMOTE_HOME/cv-singleline-processor-CV-1757/scripts/golden_dataset/streamlit_expected_sku_review.py"
```

Then on the host, restart the Streamlit process that serves the annotation UI (exact command depends how it was started on that box). Interns’ saved labels in `golden_sku_truth.csv` / assignment JSON must stay untouched.

### B) TorchServe detectors

The checked-in deploy helper `scripts/deploy_gpu_5090_updated.sh` targets the **5090** host (`172.16.20.108`) and Blackwell/`sm_120` images. A **2080** box may need different CUDA/TorchServe base images — do not assume the 5090 Dockerfiles work unchanged.

On Ant-PC-2080, after cloning/syncing the detector dirs:

```bash
# On the remote host, from the repo root (only if Docker + NVIDIA runtime are ready):
# ./scripts/deploy_gpu_5090_updated.sh   # rename/adapt for 2080 before relying on it
nvidia-smi
docker info | head
```

Validate GPU visibility inside Docker before long builds. Prefer adapting ports/image names so you do not collide with any services already bound on `.100`.

---

## 3. Suggested workflow on work laptop

1. Connect to corporate VPN.
2. `ssh Ant-PC-2080` and confirm login.
3. Locate the existing `HomeDepotCV` checkout (or clone from GitHub).
4. Sync or pull **code** changes only.
5. Restart the service you care about (Streamlit and/or TorchServe).
6. Smoke-test:
   - Annotation: enter a SKU like `12XX34` (6 chars with placeholders) and confirm Save accepts it.
   - TorchServe (if deployed): `curl` the management health / inference endpoints for that host’s ports.

---

## 4. Security notes

- `*.pem` is in `.gitignore`. Keep keys out of git and chat logs.
- Prefer `chmod 400` on the private key.
- This key authenticates as **`vaibhav.singh`** — use only if you are authorized to that account on `.100`.
- For the **5090** annotation host (`.108`), continue using the existing Avinash key / docs unless access is moved.

---

## 5. Quick reference

```bash
# Laptop → Ant-PC-2080
ssh Ant-PC-2080

# From Cursor/home Mac without VPN: will fail (expected)
# ping 172.16.20.100  → 100% loss until on corp network
```

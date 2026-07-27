# Deploy Streamlit review app to RTX 5090 host

Shared team URL (after deploy, on VPN): **http://172.16.20.108:8503**

Share that URL only with teammates who are connected to the LF/corporate VPN.
Each reviewer enters only their name. New names are registered automatically and
the remaining unstarted images are rebalanced by SKU-crop count across everyone
who has joined. Every crop from one source image always stays with one reviewer,
and images with saved reviews remain pinned to their existing reviewer.

## 1. Check access

Connect to LF/VPN first, then:

```bash
chmod +x scripts/deploy/check_gpu_host_access.sh
./scripts/deploy/check_gpu_host_access.sh
```

The documented 5090 machine is **`172.16.20.108`** (`GPU5-A5090`). If your team uses a different host ending in `.104`, the check script tries **`172.16.20.104`** as well.

## 2. Deploy

```bash
chmod +x scripts/deploy/deploy_streamlit_to_gpu_host.sh
./scripts/deploy/deploy_streamlit_to_gpu_host.sh
```

Optional alternate host:

```bash
./scripts/deploy/deploy_streamlit_to_gpu_host.sh --host 172.16.20.104
```

This syncs:

- `scripts/golden_dataset/streamlit_expected_sku_review.py` and helpers
- `golden_sku_truth.csv`, crop images, overlay images
- Creates/updates `~/HomeDepotCV/.venv` on the server
- Installs a **systemd** service (`streamlit-golden-sku`) bound to `0.0.0.0:8503`

## 3. After you upload new images

On the server (or rsync from Mac):

1. Put new image/json pairs under `~/HomeDepotCV/Golden_Dataset_overhead_eval_expected_sku/`
2. Regenerate crops and truth template:

```bash
cd ~/HomeDepotCV/cv-singleline-processor-CV-1757
source ../.venv/bin/activate
python scripts/golden_dataset/run_golden_dataset_local.py --mode crops --save-crops
python scripts/golden_dataset/generate_golden_label_overlays.py
# regenerate golden_sku_truth.csv if you add a generator step
```

3. Restart Streamlit:

```bash
sudo systemctl restart streamlit-golden-sku
```

## 4. Useful commands on the server

```bash
sudo systemctl status streamlit-golden-sku
sudo journalctl -u streamlit-golden-sku -f
sudo systemctl restart streamlit-golden-sku
```

## Notes

- Streamlit does **not** need a GPU; the 5090 host is used as a shared always-on Linux box on the team network.
- All reviewers share one `golden_sku_truth.csv`. Saves are row-level, locked, and
  atomic so reviewers working concurrently do not replace each other's rows.
- Closing the browser loses only an unsaved edit. On reopen, enter the same name;
  the default `Unreviewed` filter resumes at that reviewer's first unfinished crop.
- If port 8503 is blocked externally, ask infra to allow TCP 8503 to the host from
  the corporate network.

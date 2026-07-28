# Dual TorchServe — quick run guide (GPU1-A2080)

## Steps (overview)

1. Build / start the dual container  
2. Health-check both models  
3. Run detection on a full shelf image  
4. Run segmentation on strip crop(s)  
5. (Optional) Second detection image  

**Remember:** detector and segmenter are two endpoints on one container — not one combined response.  
**Repo:** `/data/avinash.patel/HomeDepotCV`  
**Endpoints:** `/predictions/detector` and `/predictions/segmenter` on port `9000`

---

## Paste-all commands

```bash
# === Dual TorchServe: build, health, detect, (optional) segment ===

# 0) Always start from repo root
cd /data/avinash.patel/HomeDepotCV
git pull origin main

# If Docker build previously failed with "no space left on device":
# docker builder prune -af
# docker image prune -af
# df -h

# 1) Build / start (needs best.pt + segmentation.pt)
./cv-singleline-torchserve-dual/scripts/build_and_run.sh

# 2) Health
curl -s http://127.0.0.1:9000/ping; echo
curl -s http://127.0.0.1:9001/models/detector | python3 -c "import sys,json; d=json.load(sys.stdin); print('detector', d[0]['workers'][0]['status'])"
curl -s http://127.0.0.1:9001/models/segmenter | python3 -c "import sys,json; d=json.load(sys.stdin); print('segmenter', d[0]['workers'][0]['status'])"

# 3) Detection (full shelf images — use bare filenames inside this folder)
cd /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
ls -lh test_image.jpg test_img_new.jpg
python3 test_torchserve.py test_image.jpg

# 4) Segmentation (STRIP crops only — not full shelf photos)
# Uncomment when you have a strip file:
# cd /data/avinash.patel/HomeDepotCV/cv-singleline-detector-yolov7-seg
# python3 test_torchserve.py strip_0.jpg

# 5) Optional second detection image
python3 test_torchserve.py test_img_new.jpg

echo "=== DONE ==="
echo "Detector: POST http://127.0.0.1:9000/predictions/detector"
echo "Segmenter: POST http://127.0.0.1:9000/predictions/segmenter"
```

---

## Client flow (no code)

1. `POST /predictions/detector` with full image → boxes `[x1,y1,x2,y2,conf,class]`  
2. Crop strips from the shelf / boxes  
3. `POST /predictions/segmenter` with strip(s) → masks  

Legacy `/predictions/yolov7` is for the old two-container setup only.

# Run the consolidated TorchServe service

Use these commands on the GPU host from:

```text
/data/vaibhav.singh/SingleLine_deployment
```

## 1. Check the model weights

```bash
cd /data/vaibhav.singh/SingleLine_deployment

ls -lh cv-singleline-torchserve-dual/detection/best.pt
ls -lh cv-singleline-torchserve-dual/segmentation/segmentation.pt
```

Do not build until both files exist.

## 2. Build and start

```bash
./cv-singleline-torchserve-dual/scripts/build_and_run.sh
```

The helper builds from `cv-singleline-torchserve-dual/`, creates separate MAR
files, starts `hd-dual-gpu`, and waits for TorchServe health.

## 3. Check both workers

```bash
curl -s http://127.0.0.1:9000/ping; echo
curl -s http://127.0.0.1:9001/models/detector | python3 -m json.tool
curl -s http://127.0.0.1:9001/models/segmenter | python3 -m json.tool
```

Expected:

- `/ping` reports `Healthy`
- detector worker reports `READY`
- segmenter worker reports `READY`

## 4. Test detection

```bash
cd /data/vaibhav.singh/SingleLine_deployment/cv-singleline-torchserve-dual/detection
python3 test_torchserve.py
```

This sends `test-fixtures/detection/test_image.jpg` to:

```text
POST http://127.0.0.1:9000/predictions/detector
```

## 5. Test segmentation

```bash
cd /data/vaibhav.singh/SingleLine_deployment/cv-singleline-torchserve-dual/segmentation
python3 test_torchserve.py
```

This sends `test-fixtures/segmentation/strip_0.jpg` to:

```text
POST http://127.0.0.1:9000/predictions/segmenter
```

Test all saved strips:

```bash
python3 test_torchserve.py \
  ../test-fixtures/segmentation-comparison/strips/strip_0.jpg \
  ../test-fixtures/segmentation-comparison/strips/strip_1.jpg \
  ../test-fixtures/segmentation-comparison/strips/strip_2.jpg \
  ../test-fixtures/segmentation-comparison/strips/strip_3.jpg
```

## Standalone builds

These are retained for split-container testing:

```bash
docker build -t hd-det-gpu \
  cv-singleline-torchserve-dual/detection

docker build -t hd-seg-gpu \
  cv-singleline-torchserve-dual/segmentation
```

The standalone endpoint name remains `/predictions/yolov7`. Override a test
script's endpoint with `TORCHSERVE_URL`.

## Logs

```bash
docker logs --tail 100 hd-dual-gpu
```

The old top-level detector directory names are no longer used.

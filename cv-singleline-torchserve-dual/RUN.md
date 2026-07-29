# Run the dual TorchServe service

Use this guide on `GPU1-A2080`. It starts one container that exposes detection
and segmentation as two endpoints on port `9000`.

The detector accepts full shelf images. The segmenter accepts shelf-strip
crops. Calling the detector does not automatically call the segmenter.

## 1. Go to the repository

```bash
cd /data/$USER/HomeDepotCV
git pull origin main
```

The Docker build must run from the repository root because it needs the source
code and weights from both model projects.

## 2. Confirm that both model weights exist

```bash
ls -lh cv-singleline-detector-yolo7_det_dep_2/best.pt
ls -lh cv-singleline-detector-yolov7-seg/segmentation.pt
```

Do not continue if either command reports that the file is missing.

## 3. Build and start the container

```bash
./cv-singleline-torchserve-dual/scripts/build_and_run.sh
```

This command:

1. builds separate detection and segmentation model archives
2. builds the `hd-dual-gpu` Docker image
3. removes old `hd-dual-gpu`, `hd-det-gpu`, and `hd-seg-gpu` containers
4. starts the new container
5. waits for TorchServe to report that it is healthy

The first build can take several minutes because it installs the CUDA, PyTorch,
TorchServe, and model dependencies.

## 4. Check both model workers

```bash
curl -s http://127.0.0.1:9000/ping; echo

curl -s http://127.0.0.1:9001/models/detector \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('detector:', d[0]['workers'][0]['status'])"

curl -s http://127.0.0.1:9001/models/segmenter \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('segmenter:', d[0]['workers'][0]['status'])"
```

Expected output:

```text
{"status":"Healthy"}
detector: READY
segmenter: READY
```

`Healthy` means TorchServe is running. `READY` confirms that each model has
loaded in its own worker.

## 5. Test detection with a full shelf image

```bash
cd /data/$USER/HomeDepotCV/cv-singleline-detector-yolo7_det_dep_2
python3 test_torchserve.py test_image.jpg
```

The test sends the image to:

```text
POST http://127.0.0.1:9000/predictions/detector
```

The response contains detections in the form
`[x1, y1, x2, y2, confidence, class]`.

To test another full image:

```bash
python3 test_torchserve.py test_img_new.jpg
```

## 6. Test segmentation with a shelf-strip crop

```bash
cd /data/$USER/HomeDepotCV/cv-singleline-detector-yolov7-seg
python3 test_torchserve.py strip_0.jpg
```

The test sends the strip to:

```text
POST http://127.0.0.1:9000/predictions/segmenter
```

Do not use a full shelf image for this test. The segmentation model expects
narrow strip crops. Multiple strips can be sent in one request:

```bash
python3 test_torchserve.py strip_0.jpg strip_1.jpg
```

## Normal application flow

1. Send the full image to `/predictions/detector`.
2. Use the detections to prepare the shelf-strip crops required by the
   application.
3. Send the strips to `/predictions/segmenter`.
4. Combine or store the two results in the calling application.

The old `/predictions/yolov7` endpoint belongs to the legacy two-container
deployment and should not be used with this container.

## Memory footprint

No measured VRAM/RAM figure is recorded in this repo. Dual mode loads
**detector + segmenter on one GPU** with **1 worker per model**. Idle load is
typically several GB of VRAM; peak is higher during inference. Host RAM for
TorchServe is separate from GPU memory.

Check while the container is up:

```bash
docker stats hd-dual-gpu --no-stream
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Run once after both workers are `READY`, and again during a detect/segment
call, if you need capacity numbers. On OOM, keep 1 worker/model and lower
concurrency.

## If startup fails

Check the container logs:

```bash
docker logs --tail 100 hd-dual-gpu
```

If Docker reports `no space left on device`, inspect disk usage before removing
unused build data:

```bash
df -h
docker system df
```

More setup and troubleshooting details are in
`TORCHSERVE_SSH_DEPLOY_172.16.20.100.md`.

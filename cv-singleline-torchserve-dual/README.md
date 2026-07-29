# Detection and segmentation in one TorchServe container

This folder is the deployment layer that runs the existing detection and
segmentation models together on one GPU server. It does not contain a third
model or replace either YOLO codebase.

For copy-and-paste commands, see [RUN.md](RUN.md).

## The problem this solves

The original deployment ran two Docker containers:

- detection on port `9000`
- segmentation on port `10000`

That worked, but it meant building, starting, monitoring, and connecting to two
separate services.

Putting both YOLO projects directly into one Python environment was not safe.
The projects use some of the same import names, such as `models` and `utils`,
but those names refer to different code. If both source trees share one Python
path, one model can import files from the other project and fail at startup or
run the wrong code.

## The solution

One container now runs one TorchServe service with two model packages:

- `detector.mar` contains the detection weights and only the `yolov7/` code.
- `segmenter.mar` contains the segmentation weights and only the
  `yolov7-seg/` code.

TorchServe unpacks each package into its own model directory and starts each
model in a separate worker process. This keeps their Python imports isolated
while still allowing both workers to use the same GPU.

The two API endpoints are:

- `POST /predictions/detector` for a full shelf image
- `POST /predictions/segmenter` for one or more shelf-strip crops

Both endpoints use port `9000`. They are separate requests; the container does
not automatically pass detector output to the segmenter.

## Request flow

1. Send a full shelf image to the detector.
2. The detector returns product/shelf bounding boxes.
3. The calling application creates the required shelf-strip crops.
4. Send those strips to the segmenter.
5. The segmenter returns the segmentation results.

The application remains responsible for steps 3 and 4.

## What improves

- There is one image and one container to deploy instead of two.
- Clients use one base URL and select the model by endpoint.
- Each model still uses its own source tree, configuration, weights, and worker.
- Import-name collisions between the two YOLO projects are avoided.
- A failure or code change in one handler is less likely to affect the other
  model's imports.

This changes how the models are packaged and served. It does not change model
accuracy, combine the two responses, or remove the GPU memory needed to load
both models.

## Memory footprint

There is no checked-in measured VRAM/RAM number for this dual container. Both
models load onto the **same GPU** with **one TorchServe worker each**
(`min_workers=1` / `max_workers=1` in `config.properties`), which is the
intended layout for an RTX 2080-class card.

Expect several GB of GPU memory for idle load (two YOLOv7 weights + CUDA /
TorchServe overhead). Usage rises during inference. Host RAM for the Java /
TorchServe process is separate from GPU VRAM.

Measure on the GPU while `hd-dual-gpu` is running:

```bash
docker stats hd-dual-gpu --no-stream
nvidia-smi

# After detect + segment calls, check again
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Capture `nvidia-smi` once both workers are `READY`, then again mid-request, if
you need numbers for capacity planning. If both models OOM on the 2080, keep
one worker per model, reduce concurrent requests, and re-check `nvidia-smi`.

## What happens during build and startup

The build uses files from all three project folders:

1. `best.pt` and `yolov7/` are copied from
   `cv-singleline-detector-yolo7_det_dep_2`.
2. `segmentation.pt` and `yolov7-seg/` are copied from
   `cv-singleline-detector-yolov7-seg`.
3. This folder adds separate handlers and configuration for each model.
4. The files are archived as `detector.mar` and `segmenter.mar`.
5. TorchServe starts one worker for each archive.

The helper script checks that both weight files exist, builds the Docker image,
removes old detection/segmentation containers, starts the new container, and
waits for TorchServe to become healthy.

## Build and start

Run this from the repository root on `GPU1-A2080`:

```bash
cd /data/$USER/HomeDepotCV

ls cv-singleline-detector-yolo7_det_dep_2/best.pt
ls cv-singleline-detector-yolov7-seg/segmentation.pt

./cv-singleline-torchserve-dual/scripts/build_and_run.sh
```

The server ports are:

- `9000`: inference requests
- `9001`: model management and worker status
- `9002`: metrics

## Check the service

```bash
curl -s http://127.0.0.1:9000/ping
curl -s http://127.0.0.1:9001/models/detector
curl -s http://127.0.0.1:9001/models/segmenter
```

`/ping` should return `{"status":"Healthy"}`. Each model status should show a
worker with status `READY`.

To test the full service:

```bash
cd /data/$USER/HomeDepotCV
python3 scripts/smoke_test_gpu_detectors_updated.py \
  --base-url http://127.0.0.1:9000
```

For first-time server setup, weight downloads, SSH instructions, and
troubleshooting, see `TORCHSERVE_SSH_DEPLOY_172.16.20.100.md`.

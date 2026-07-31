# Detection model

This directory contains only the full-shelf detection model and its deployment
code.

## Contents

- `best.pt`: model weights (not committed to Git)
- `yolov7/`: detection model source
- `service_pipeline_gpu/stage1_detection.py`: inference pipeline
- `model_handler.py`: standalone TorchServe detection handler
- `Dockerfile`: optional standalone image
- `model.sh`: downloads `best.pt`

There is no segmentation model, segmentation weight, or `yolov7-seg` source in
this directory.

## Restore weights

```bash
cd cv-singleline-torchserve-dual/detection
bash model.sh
```

## Standalone build

```bash
docker build -t hd-det-gpu cv-singleline-torchserve-dual/detection
```

The standalone model uses `/predictions/yolov7`. The preferred dual deployment
uses `/predictions/detector`.

## Test the dual endpoint

```bash
cd cv-singleline-torchserve-dual/detection
python3 test_torchserve.py
```

The default image is stored in `../test-fixtures/detection/`.

Files retained only for reference are grouped under `dev/`.

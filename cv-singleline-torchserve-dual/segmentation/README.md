# Segmentation model

This directory contains only the shelf-strip segmentation model and its
deployment code.

## Contents

- `segmentation.pt`: model weights (not committed to Git)
- `yolov7-seg/`: segmentation model source
- `service_pipeline_gpu/stage2_segmentation.py`: inference pipeline
- `model_handler.py`: standalone TorchServe segmentation handler
- `Dockerfile`: optional standalone image
- `model.sh`: downloads `segmentation.pt`

There is no detection weight, detection stage, or `yolov7` detection source in
this directory.

## Restore weights

```bash
cd cv-singleline-torchserve-dual/segmentation
bash model.sh
```

## Standalone build

```bash
docker build -t hd-seg-gpu cv-singleline-torchserve-dual/segmentation
```

The standalone model uses `/predictions/yolov7`. The preferred dual deployment
uses `/predictions/segmenter`.

## Test the dual endpoint

```bash
cd cv-singleline-torchserve-dual/segmentation
python3 test_torchserve.py
```

The default strip and previous/new comparison masks are stored under
`../test-fixtures/`.

Files retained only for reference are grouped under `dev/`.

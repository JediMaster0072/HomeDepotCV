#!/bin/bash
set -e
echo "Downloading YOLOv7 weights from GCS..."
# gsutil cp gs://selling-pipeline-ml-models/multiline-pipeline-ml-models/singleline-pipeline-seg/best.pt .
gsutil cp gs://selling-pipeline-ml-models/singleline-pipeline-seg-models-1.1/segmentation.pt .
echo "Downloaded best.pt successfully"
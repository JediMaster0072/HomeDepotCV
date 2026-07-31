#!/bin/bash
set -e
echo "Downloading YOLOv7 weights from GCS..."
gsutil cp gs://selling-pipeline-ml-models/singleline-pipeline-det-models-1.2/best.pt .
echo "Downloaded best.pt successfully"
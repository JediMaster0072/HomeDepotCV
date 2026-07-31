#!/bin/bash
set -e
echo "Downloading YOLOv7-seg weights from GCS..."
gsutil cp gs://selling-pipeline-ml-models/singleline-pipeline-seg-models-1.1/segmentation.pt .
echo "Downloaded segmentation.pt successfully"
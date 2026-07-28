# Dual TorchServe image — one process, two MAR workers (detector + segmenter).
#
# Build from HomeDepotCV repo root:
#   docker build -f cv-singleline-torchserve-dual/Dockerfile -t hd-dual-gpu .
#
# Or:
#   ./cv-singleline-torchserve-dual/scripts/build_and_run.sh

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
# Do NOT put YOLO source trees on PYTHONPATH — each worker uses its MAR model_dir only.
ENV PYTHONPATH=

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    openjdk-17-jre-headless \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    rsync \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip3 install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install torchserve torch-model-archiver

COPY cv-singleline-torchserve-dual/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN pip3 install \
    filterpy \
    scikit-image \
    onnx \
    onnxruntime \
    future \
    packaging

# --------------------------------------------------------------------
# Stage MAR build inputs (separate trees — never merge yolov7 + yolov7-seg)
# --------------------------------------------------------------------
WORKDIR /mar_build

# Detection staging
RUN mkdir -p /mar_build/detection
COPY cv-singleline-detector-yolo7_det_dep_2/best.pt /mar_build/detection/best.pt
COPY cv-singleline-detector-yolo7_det_dep_2/yolov7 /mar_build/detection/yolov7
COPY cv-singleline-detector-yolo7_det_dep_2/service_pipeline_gpu /mar_build/detection/service_pipeline_gpu
COPY cv-singleline-detector-yolo7_det_dep_2/codec_gpu.py /mar_build/detection/codec_gpu.py
COPY cv-singleline-torchserve-dual/packaging/detection/common_config_gpu.py /mar_build/detection/common_config_gpu.py
COPY cv-singleline-torchserve-dual/handlers/detection_handler.py /mar_build/detection/model_handler.py

# Segmentation staging
RUN mkdir -p /mar_build/segmentation
COPY cv-singleline-detector-yolov7-seg/segmentation.pt /mar_build/segmentation/segmentation.pt
COPY cv-singleline-detector-yolov7-seg/yolov7-seg /mar_build/segmentation/yolov7-seg
COPY cv-singleline-detector-yolov7-seg/service_pipeline_gpu /mar_build/segmentation/service_pipeline_gpu
COPY cv-singleline-detector-yolov7-seg/codec_gpu.py /mar_build/segmentation/codec_gpu.py
COPY cv-singleline-torchserve-dual/packaging/segmentation/common_config_gpu.py /mar_build/segmentation/common_config_gpu.py
COPY cv-singleline-torchserve-dual/handlers/segmentation_handler.py /mar_build/segmentation/model_handler.py

# Drop wandb / caches from staged trees (shrink image, avoid noise)
RUN find /mar_build -type d \( -name wandb -o -name __pycache__ -o -name .git \) -prune -exec rm -rf {} + 2>/dev/null || true

# Build detector.mar (yolov7 only).
# Zip extras so directory layout (service_pipeline_gpu/, yolov7/) is preserved in the MAR.
WORKDIR /mar_build/detection
RUN zip -r /mar_build/det_extra.zip yolov7 service_pipeline_gpu codec_gpu.py common_config_gpu.py \
    && torch-model-archiver \
    --model-name detector \
    --version 1.0 \
    --handler model_handler.py \
    --serialized-file best.pt \
    --extra-files /mar_build/det_extra.zip \
    --export-path /mar_build \
    --force

# Build segmenter.mar (yolov7-seg only)
WORKDIR /mar_build/segmentation
RUN zip -r /mar_build/seg_extra.zip yolov7-seg service_pipeline_gpu codec_gpu.py common_config_gpu.py \
    && torch-model-archiver \
    --model-name segmenter \
    --version 1.0 \
    --handler model_handler.py \
    --serialized-file segmentation.pt \
    --extra-files /mar_build/seg_extra.zip \
    --export-path /mar_build \
    --force

# --------------------------------------------------------------------
# Runtime image contents: config + model_store only (no shared YOLO PYTHONPATH)
# --------------------------------------------------------------------
WORKDIR ${APP_HOME}
RUN mkdir -p ${APP_HOME}/model_store \
    && mv /mar_build/detector.mar ${APP_HOME}/model_store/ \
    && mv /mar_build/segmenter.mar ${APP_HOME}/model_store/ \
    && rm -rf /mar_build

COPY cv-singleline-torchserve-dual/config.properties ${APP_HOME}/config.properties

RUN python3 - <<'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available (build-time):", torch.cuda.is_available())
EOF

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--foreground", "--ncs", \
     "--ts-config", "/app/config.properties", \
     "--model-store", "/app/model_store", \
     "--models", "detector=detector.mar,segmenter=segmenter.mar"]

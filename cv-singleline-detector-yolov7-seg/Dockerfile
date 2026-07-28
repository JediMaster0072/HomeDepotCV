FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PYTHONPATH=${APP_HOME}
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR ${APP_HOME}

# --------------------------------------------------------------------
# System packages
# --------------------------------------------------------------------
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
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------------------------
# CUDA 12.8 PyTorch
# --------------------------------------------------------------------
RUN pip3 install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# --------------------------------------------------------------------
# TorchServe
# --------------------------------------------------------------------
RUN pip3 install \
    torchserve \
    torch-model-archiver

# --------------------------------------------------------------------
# Project requirements (DO NOT use --no-deps; it breaks SciPy wheels)
# --------------------------------------------------------------------
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install \
    requests \
    urllib3 \
    idna \
    certifi \
    charset-normalizer \
    filterpy \
    scikit-image \
    onnx \
    onnxruntime \
    future \
    packaging

COPY . .

RUN python3 - <<'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available (build-time):", torch.cuda.is_available())
EOF

# --------------------------------------------------------------------
# Build model archive
# --------------------------------------------------------------------
RUN torch-model-archiver \
    --model-name yolov7 \
    --version 1.0 \
    --handler model_handler.py \
    --serialized-file segmentation.pt \
    --extra-files "yolov7-seg/,service_pipeline_gpu/,codec_gpu.py,common_config_gpu.py" \
    --export-path .

RUN mkdir -p model_store && mv yolov7.mar model_store/

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--foreground", "--ncs", "--ts-config", "/app/config.properties", "--model-store", "model_store", "--models", "yolov7=yolov7.mar"]

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
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

# --------------------------------------------------------------------
# Upgrade pip
# --------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------------------------
# Install CUDA 12.8 PyTorch
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
# Copy requirements
# --------------------------------------------------------------------
COPY requirements.txt .

# --------------------------------------------------------------------
# Install project requirements
# (DO NOT USE --no-deps)
# --------------------------------------------------------------------
RUN pip3 install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------
# Install additional packages frequently required by YOLOv7/TorchServe
# --------------------------------------------------------------------
RUN pip3 install \
    requests \
    urllib3 \
    idna \
    certifi \
    charset-normalizer \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    pillow \
    opencv-python \
    pyyaml \
    tqdm \
    onnx \
    onnxruntime \
    filterpy \
    scikit-image \
    protobuf \
    psutil \
    future \
    packaging \
    python-dateutil \
    pytz \
    grpcio \
    markdown \
    tensorboard \
    tensorboard-data-server \
    werkzeug

# --------------------------------------------------------------------
# Copy project
# --------------------------------------------------------------------
COPY . .

# --------------------------------------------------------------------
# Verify imports BEFORE runtime
# --------------------------------------------------------------------
RUN python3 - <<'EOF'
import importlib

mods = [
    "torch",
    "torchvision",
    "torchaudio",
    "ts",
    "cv2",
    "numpy",
    "PIL",
    "yaml",
    "requests",
    "urllib3",
    "idna",
    "certifi",
    "charset_normalizer",
    "onnx",
    "onnxruntime",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "skimage",
    "filterpy",
    "tqdm",
]

missing = []

for m in mods:
    try:
        importlib.import_module(m)
        print(f"[OK] {m}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
        missing.append(m)

if missing:
    raise RuntimeError(f"Missing modules: {missing}")

print("All required Python modules imported successfully.")
EOF

# --------------------------------------------------------------------
# Verify CUDA
# --------------------------------------------------------------------
RUN python3 - <<'EOF'
import torch

print("Torch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# --------------------------------------------------------------------
# Build TorchServe model archive
# --------------------------------------------------------------------
RUN torch-model-archiver \
    --model-name yolov7 \
    --version 1.0 \
    --handler model_handler.py \
    --serialized-file best.pt \
    --extra-files "yolov7/,service_pipeline_gpu/,codec_gpu.py,common_config_gpu.py" \
    --export-path .

# --------------------------------------------------------------------
# Model store
# --------------------------------------------------------------------
RUN mkdir -p model_store && mv yolov7.mar model_store/

# --------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------
EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

# --------------------------------------------------------------------
# Launch TorchServe
# --------------------------------------------------------------------
CMD ["torchserve", "--start", "--foreground", "--ncs", "--ts-config", "/app/config.properties", "--model-store", "model_store", "--models", "yolov7=yolov7.mar"]

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/app
ENV PYTHONPATH=${APP_HOME}

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir -r requirements.txt --no-deps

COPY . .

RUN python3 -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.version.cuda)"

RUN torch-model-archiver \
    --model-name yolov7 \
    --version 1.0 \
    --handler model_handler.py \
    --extra-files "common_config_gpu.py,codec_gpu.py,service_pipeline_gpu"

RUN mkdir modelstore && mv yolov7.mar modelstore/

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--ncs", "--ts-config", "/app/config.properties", "--model-store", "modelstore", "--models", "yolov7.mar"]

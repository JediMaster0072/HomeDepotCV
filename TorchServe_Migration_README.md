# Home Depot Single-Line Detector

## TorchServe Migration Guide

> **Purpose:** This document explains what this project is, why
> TorchServe is used, how the repository is organized, and how to build,
> run, test, and troubleshoot the application.

------------------------------------------------------------------------

# Quick Start

1.  SSH into the correct GPU machine.

2.  Verify the GPU is available:

    ``` bash
    nvidia-smi
    ```

3.  Build the Docker image.

4.  Run the Docker container.

5.  Verify TorchServe is healthy.

6.  Send the example request (`request.json`).

7.  Confirm detections are returned.

------------------------------------------------------------------------

# 1. Project Overview

This project serves a YOLOv7 object detection model through TorchServe.

Instead of running Python scripts manually, TorchServe keeps the model
loaded in memory and waits for requests. When an image is sent,
TorchServe passes it to the model and returns the detection results as
JSON.

### Why was this migration done?

-   Easier deployment
-   Standard REST API
-   Better reliability
-   Easier scaling
-   Easier monitoring

------------------------------------------------------------------------

# 2. What is TorchServe?

TorchServe is a model serving framework for PyTorch.

Think of it as a web server designed specifically for AI models.

It: - Starts the model - Keeps it in memory - Receives HTTP requests -
Runs inference - Returns results

Request flow:

``` text
Client
  │
HTTP Request
  │
TorchServe
  │
model_handler.py
  │
preprocess()
  │
YOLOv7
  │
postprocess()
  │
JSON Response
```

------------------------------------------------------------------------

# 3. Repository Structure

  -----------------------------------------------------------------------
  File                         Purpose
  ---------------------------- ------------------------------------------
  Dockerfile                   Builds the Docker image.

  config.properties            TorchServe configuration (ports, workers,
                               models).

  model_handler.py             Main inference logic.

  best.pt                      Trained YOLOv7 weights.

  request.json                 Example inference request containing a
                               Base64 image.

  model_store/                 Stores the `.mar` model archive.

  test_image.jpg               Example image for testing.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 4. Machine Requirements

-   Ubuntu Linux
-   NVIDIA GPU
-   NVIDIA Driver
-   CUDA
-   Docker
-   NVIDIA Container Toolkit

Verify the GPU:

``` bash
nvidia-smi
```

------------------------------------------------------------------------

# 5. Build the Docker Image

``` bash
docker build -t hd-det-gpu .
```

This installs all required software and packages the model into a
reusable image.

------------------------------------------------------------------------

# 6. Run the Container

Example:

``` bash
docker run \
  --gpus all \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  --name hd-det-test \
  hd-det-gpu
```

Important flags:

-   `--gpus all` → allow Docker to use the GPU.
-   `8080` → inference API.
-   `8081` → management API.
-   `8082` → metrics API.

------------------------------------------------------------------------

# 7. Verify Everything Started

Health:

``` bash
curl http://localhost:8080/ping
```

Expected:

``` json
{"status":"Healthy"}
```

Loaded models:

``` bash
curl http://localhost:8081/models
```

Expected to see `yolov7`.

------------------------------------------------------------------------

# 8. Running Inference

The handler expects **JSON**, not a raw image.

Correct:

``` bash
curl \
-X POST \
http://localhost:8080/predictions/yolov7 \
-H "Content-Type: application/json" \
--data-binary @request.json
```

Do **not** send:

``` bash
curl -T image.jpg
```

That sends raw bytes and will fail.

------------------------------------------------------------------------

# 9. Understanding the Output

Each detection looks like:

``` text
[xmin, ymin, xmax, ymax, confidence, class_id]
```

Example:

``` text
[4304,92,4835,407,0.88,0]
```

Meaning:

-   Bounding box coordinates
-   88% confidence
-   Class 0

------------------------------------------------------------------------

# 10. Common Issues

## preprocessing failed

Cause:

A raw image was sent instead of JSON.

Fix:

Use `request.json`.

------------------------------------------------------------------------

## bytearray indices must be integers

Cause:

The handler expected:

``` python
request[0]["body"]["instances"]
```

but received raw image bytes.

------------------------------------------------------------------------

## Model does not load

Check:

-   Docker logs
-   `config.properties`
-   `.mar` file
-   `model_store`

------------------------------------------------------------------------

## GPU not found

Run:

``` bash
nvidia-smi
```

------------------------------------------------------------------------

# 11. Useful Commands

``` bash
docker ps
docker logs hd-det-test
docker images
docker stop hd-det-test
docker rm hd-det-test
```

TorchServe:

``` bash
curl http://localhost:8080/ping
curl http://localhost:8081/models
```

Linux:

``` bash
pwd
ls
find
```

------------------------------------------------------------------------

# 12. Lessons Learned

-   TorchServe changes the request format depending on the content type.
-   This handler expects JSON with a Base64 image.
-   `/ping` checks the server.
-   `/models` confirms the model loaded.
-   A successful prediction confirms the full pipeline is working.
-   Each worker loads its own copy of the model.

------------------------------------------------------------------------

# Future Improvements

-   Add architecture diagrams.
-   Explain the `.mar` build process.
-   Document the Dockerfile line by line.
-   Add deployment instructions for new GPU servers.
-   Add performance tuning guidance.

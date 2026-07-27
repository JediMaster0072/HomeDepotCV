# Why We Need a New Blackwell-Compatible TorchServe Base Image

## Current Situation

Our project was originally built on:

-   `pytorch/torchserve:0.12.0-gpu` (detection)
-   `pytorch/torchserve:0.8.1-gpu` (segmentation)

These images include: - TorchServe - `torch-model-archiver` - Python
environment - PyTorch - CUDA runtime

They worked correctly on older NVIDIA GPUs.

## What Changed

The deployment target is now an NVIDIA RTX 5090 (Blackwell).

The installed PyTorch inside the existing TorchServe images only
supports GPU architectures up to `sm_90`.

The RTX 5090 reports:

    sm_120

During verification we saw:

    AssertionError: sm_120 NOT supported

This means the installed PyTorch cannot generate kernels for the GPU.

## Why the Simple Upgrade Failed

Replacing the base image with:

    FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

gave us a modern CUDA environment, but removed TorchServe because it is
not part of NVIDIA CUDA images.

As a result:

    torch-model-archiver: not found

The build failed before packaging the model.

## Why We Shouldn't Only Upgrade PyTorch

TorchServe images are built and tested around specific PyTorch versions.

Replacing only PyTorch can introduce compatibility issues between: -
TorchServe - model handlers - packaging tools - serving runtime

Even if the build succeeds, runtime failures become more likely.

## Recommended Approach

Create a new TorchServe base image that already contains: - CUDA 12.8+ -
A Blackwell-compatible PyTorch release - TorchServe -
torch-model-archiver

Then update the project Dockerfiles to inherit from this new base image.

This preserves nearly all existing Dockerfile logic while adding RTX
5090 support.

## Benefits

-   Preserves the existing deployment workflow.
-   Keeps TorchServe and model packaging intact.
-   Supports RTX 5090 (`sm_120`).
-   Minimizes changes to the project.
-   Easier to maintain and upgrade in the future.

## Summary

The problem is not Docker or the NVIDIA driver. Those have already been
verified.

The remaining incompatibility is the software stack inside the
TorchServe base images. Building a modern TorchServe base image with a
Blackwell-compatible PyTorch is the cleanest and most maintainable
solution.

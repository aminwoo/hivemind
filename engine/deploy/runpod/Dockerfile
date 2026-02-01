# Hivemind Bughouse Engine - RunPod Serverless Deployment
# Based on NVIDIA TensorRT container for GPU inference
# Using 25.01 for TensorRT 10.8 (closest to local 10.14)

FROM nvcr.io/nvidia/tensorrt:25.01-py3

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the handler
RUN pip install --no-cache-dir \
    runpod \
    requests

# Copy engine source code
COPY src/ /app/src/
COPY cmake/ /app/cmake/
COPY tests/ /app/tests/
COPY CMakeLists.txt /app/

# Build the engine
RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" && \
    ninja

# Copy the network model (ONNX and pre-built TensorRT engine for fast startup)
RUN mkdir -p /app/networks
COPY networks/*.onnx /app/networks/
COPY networks/*_fp16_b16_gpu0_v1.engine /app/networks/

# Copy the RunPod handler
COPY deploy/runpod/handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENGINE_PATH=/app/build/hivemind
ENV NETWORKS_DIR=/app/networks

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]

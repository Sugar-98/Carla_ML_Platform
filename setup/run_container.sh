#!/bin/bash

# Specify the image base name
IMAGE_NAME="carla_ml_platform"

# Specify the tag name (e.g., date or version)
TAG_NAME="latest"

# Full image name with tag
IMAGE_NAME_WITH_TAG="${IMAGE_NAME}:${TAG_NAME}"

# Set host directory and container name (with tag suffix)
HOST_DIR="$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")" #The directory where project is cloned.  
CONTAINER_DIR="/home/workspace"
CONTAINER_NAME="${IMAGE_NAME}_${TAG_NAME}"

# Detect WSL
if grep -qi microsoft /proc/version; then
  echo "Running inside WSL — enabling WSLg mounts"
  EXTRA_WSL_OPTS="
    -v /mnt/wslg:/mnt/wslg
    -v /usr/lib/wsl:/usr/lib/wsl
    --device=/dev/dxg
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib
  "
else
  echo "Running on Linux native — skipping WSLg mounts"
  EXTRA_WSL_OPTS=""
fi

# Detect GPU availability for Docker (NVIDIA Container Toolkit)
GPU_OPTS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  # nvidia-smi exists on host → likely NVIDIA GPU/driver present
  if docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia"; then
    echo "NVIDIA runtime detected — enabling --gpus all"
    GPU_OPTS+=(--gpus all)
  else
    echo "nvidia-smi exists but Docker NVIDIA runtime not detected — running CPU mode"
  fi
else
  echo "No nvidia-smi — running CPU mode"
fi

# Run the container with all mounts and environment variables
docker run \
  "${GPU_OPTS[@]}" \
  --shm-size=8g \
  -p 8891:8888 \
  -v "$HOST_DIR":"$CONTAINER_DIR" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  $EXTRA_WSL_OPTS \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e PULSE_SERVER=$PULSE_SERVER \
  --name "$CONTAINER_NAME" \
  -it "$IMAGE_NAME_WITH_TAG"
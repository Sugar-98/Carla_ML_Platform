#!/bin/bash

# Specify the image base name
IMAGE_NAME="CARLA_ML_PLATFORM"

# Specify the tag name (e.g., date or version)
TAG_NAME="latest"

# Full image name with tag
IMAGE_NAME_WITH_TAG="${IMAGE_NAME}:${TAG_NAME}"

# Set host directory and container name (with tag suffix)
HOST_DIR="$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")" #The directory where project is cloned.  
CONTAINER_DIR="/home/workspace/LiftSplatShoot"
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


# Run the container with all mounts and environment variables
docker run --gpus all \
  --shm-size=8g \
  -p 8890:8889 \
  -v "$HOST_DIR":"$CONTAINER_DIR" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  $EXTRA_WSL_OPTS \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e PULSE_SERVER=$PULSE_SERVER \
  --name "$CONTAINER_NAME" \
  -it "$IMAGE_NAME_WITH_TAG"
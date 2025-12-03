#!/bin/bash
IMAGE_NAME="CARLA_ML_PLATFORM"
TAG="latest"

docker build -t ${IMAGE_NAME}:${TAG} .
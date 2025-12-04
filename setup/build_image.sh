#!/bin/bash
IMAGE_NAME="CARLA_ML_PLATFORM"
TAG="latest"
Dockerfile="setup.Dockerfile"

docker build -t ${IMAGE_NAME}:${TAG} -f ${Dockerfile} .
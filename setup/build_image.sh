#!/bin/bash
IMAGE_NAME="carla_ml_platform"
TAG="latest"
Dockerfile="setup.Dockerfile"

docker build -t ${IMAGE_NAME}:${TAG} -f ${Dockerfile} .
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    wget curl git unzip sudo \
    build-essential software-properties-common \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgtk-3-dev \
    libgl1 libglfw3 libglfw3-dev \
    ca-certificates \
    libfreetype6 \
    libpng16-16 \
    python3 \
    python3-pip \
    python3-dev \
    python3-distutils \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* 


# --- Upgrade pip ---
RUN python3 -m pip install --upgrade pip

# --- Copy requirement file and install ---
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8888

# ==== bashで起動 ====
CMD ["bash"]
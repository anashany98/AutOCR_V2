# Base Image: NVIDIA CUDA 12.1 with cuDNN 8 on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# - Python 3.11 (via deadsnakes PPA if needed, but 22.04 has 3.10 default. Let's stick to system python 3.10 or install 3.11 if strict.)
# Ubuntu 22.04 has Python 3.10. To match user's 3.11 environment tightly, we add deadsnakes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libmagic1 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for 3.11 manually to avoid conflicts
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

# Copy requirements first for cache efficiency
COPY requirements_docker.txt .

# Install Python dependencies
# Use --extra-index-url for PyTorch to ensure we get Linux/CUDA wheels
# --break-system-packages needed for newer pip on Ubuntu/Debian
RUN pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements_docker.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy application source code
COPY . .

# Expose server port
EXPOSE 8000

# Run the server
CMD ["python", "-u", "serve.py"]
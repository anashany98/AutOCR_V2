FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1️⃣ System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libgtk-3-0 libpango-1.0-0 libatk1.0-0 libcairo-gobject2 \
    libgdk-pixbuf-2.0-0 libjpeg-dev libopenblas-dev libstdc++6 \
    poppler-utils libgl1 tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    libssl-dev libffi-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2️⃣ Virtual env
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3️⃣ DEPENDENCIAS CON SOLUCIÓN PADDLEPADDLE
COPY requirements.txt .

# ------------------------------------------------------------
# PaddlePaddle 2.6.1 is compatible with Ubuntu 22.04's OpenSSL 3.0
# ------------------------------------------------------------

# Capa 1: NumPy y PaddlePaddle primero (orden crítico)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    paddlepaddle-gpu==2.6.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Capa 2: PyTorch CUDA 12.1 (ya funciona)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Capa 3: Resto de dependencias (sin numpy para evitar conflicto)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    paddleocr==3.3.0 \
    paddlex==3.3.0 \
    easyocr==1.7.2 \
    pytesseract>=0.3.10 \
    PyMuPDF==1.23.9 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.4.0 \
    pdf2image==1.17.0 \
    open-clip-torch==2.20.0 \
    sentencepiece==0.1.99 \
    transformers==4.30.2 \
    pyyaml>=6.0 \
    rapidfuzz==3.0.0 \
    tqdm==4.64.0 \
    pytest==7.4.0 \
    loguru==0.7.2 \
    pydantic==2.7.1 \
    waitress==3.0.0 \
    scikit-image==0.25.2 \
    shapely==2.1.2 \
    pyclipper==1.3.0.post6 \
    Flask==3.0.0 \
    flask-wtf==1.1.1 \
    wtforms==3.0.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir faiss-gpu==1.7.2

# 4️⃣ GPU verification will happen at runtime when GPUs are available

# 5️⃣ Copy app
COPY . .

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000
CMD ["python3", "run_web.py"]
# Dockerfile for Chatterbox TTS OpenAI-Compatible API
# Alternative base images - try these in order if one fails

# Option 1: CUDA 12.2 runtime (most likely to work)
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Option 2: If above fails, uncomment this and comment above
# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Option 3: If above fails, use this (older but stable)
# FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Option 4: Latest available (might be too new)
# FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-extra \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (matching your working setup)
# This should work with CUDA 12.2 base image due to compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install \
    transformers>=4.46.3 \
    diffusers>=0.29.0 \
    safetensors>=0.5.3 \
    huggingface_hub \
    accelerate

# Install audio processing dependencies
RUN pip install \
    librosa \
    resampy \
    soundfile \
    scipy \
    numpy

# Install utility dependencies
RUN pip install \
    einops \
    tqdm \
    omegaconf>=2.3.0

# Install API dependencies
RUN pip install \
    fastapi>=0.104.1 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.0.0 \
    python-multipart

# Install optional dependencies (some might fail, that's ok)
RUN pip install conformer || echo "conformer install failed"
RUN pip install s3tokenizer || echo "s3tokenizer install failed"
RUN pip install resemble-perth || echo "resemble-perth install failed"

# Try to install xformers (for better performance)
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121 || echo "xformers install failed"

# Copy the entire chatterbox project
COPY . /app/

# Add the src directory to Python path
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create a non-root user for security
RUN useradd -m -u 1000 chatterbox && chown -R chatterbox:chatterbox /app
USER chatterbox

# Expose the API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python3", "chatterbox_openai_api.py", "--host", "0.0.0.0", "--port", "8001", "--device", "cuda"]
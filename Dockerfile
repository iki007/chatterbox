# Lightweight image for the Chatterbox OpenAI-compatible API with GPU support
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# System dependencies required for audio processing and runtime healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest early to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies (CUDA wheels) and clear build caches
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip cache purge || true

# Remove build toolchain to shrink the final image
RUN apt-get purge -y --auto-remove build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/requirements.txt && \
    mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

# Copy the application source
COPY . /app/

# Ensure project modules are importable without relying on prior PYTHONPATH
ENV PYTHONPATH=/app/src

# Run as non-root user for security
RUN useradd -m -u 1000 chatterbox && chown -R chatterbox:chatterbox /app
USER chatterbox

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python3", "chatterbox_openai_api.py", "--host", "0.0.0.0", "--port", "8001", "--device", "cuda"]

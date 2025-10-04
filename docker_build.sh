#!/bin/bash
# Docker build and run helper for Chatterbox TTS

# Build script
echo "🐳 Building Chatterbox TTS Docker image..."

# Build the Docker image
docker build -t chatterbox-gpu .

echo "✅ Docker image built successfully!"

# Test that NVIDIA runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: NVIDIA Docker runtime not available"
    echo "💡 Install nvidia-docker2:"
    echo "   sudo apt-get install nvidia-docker2"
    echo "   sudo systemctl restart docker"
    exit 1
fi

echo "✅ NVIDIA Docker runtime is available"

# Ensure the host cache directory exists so the volume mount works cleanly
mkdir -p "$HOME/.cache/huggingface"

# Run script
echo "🚀 Starting Chatterbox TTS container..."

# Stop any existing container
docker stop chatterbox-tts 2>/dev/null || true
docker rm chatterbox-tts 2>/dev/null || true

# Run the container
docker run -d \
    --name chatterbox-tts \
    --gpus all \
    -e CBX_SKIP_STARTUP_LOAD=1 \
    -p 8001:8001 \
    -v "$HOME/.cache/huggingface:/home/chatterbox/.cache/huggingface" \
    --restart unless-stopped \
    --memory="16g" \
    --shm-size="2g" \
    chatterbox-gpu

echo "⏳ Waiting for container to start..."
sleep 10

# Check if container is running
if docker ps | grep -q chatterbox-tts; then
    echo "✅ Container is running!"
    echo "🌐 API available at: http://localhost:8001/v1"
    echo "📖 API docs at: http://localhost:8001/docs"
    echo "🏥 Health check: http://localhost:8001/health"
else
    echo "❌ Container failed to start. Checking logs..."
    docker logs chatterbox-tts
fi

# Test the API
echo "🧪 Testing API..."
sleep 5
curl -s http://localhost:8001/health | python3 -m json.tool || echo "❌ API not responding yet"

echo ""
echo "📋 Useful commands:"
echo "   View logs: docker logs -f chatterbox-tts"
echo "   Stop container: docker stop chatterbox-tts"
echo "   Start container: docker start chatterbox-tts"
echo "   Remove container: docker rm -f chatterbox-tts"
echo "   Shell into container: docker exec -it chatterbox-tts bash"

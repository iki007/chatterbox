#!/usr/bin/env python3
"""
Chatterbox TTS API - Compatible version without xformers
This version avoids xformers compatibility issues
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import io
import logging
import tempfile
import time
from typing import Optional, Literal

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Basic GPU optimizations (without xformers)
if torch.cuda.is_available():
    # Set memory allocation strategy for RTX 4090
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cudnn.benchmark = True
    print(f"🚀 GPU Optimizations enabled for {torch.cuda.get_device_name(0)}")
    print("ℹ️  Running without xformers - install compatible version for better performance")

# Try to import chatterbox with better error reporting
try:
    from chatterbox.tts import ChatterboxTTS
    print("✅ ChatterboxTTS imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ChatterboxTTS: {e}")
    
    # Try to identify missing dependencies
    missing_deps = []
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    try:
        import safetensors
    except ImportError:
        missing_deps.append("safetensors")
    
    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")
    
    try:
        import omegaconf
    except ImportError:
        missing_deps.append("omegaconf")
    
    try:
        import conformer
    except ImportError:
        missing_deps.append("conformer")
    
    try:
        import einops
    except ImportError:
        missing_deps.append("einops")
    
    if missing_deps:
        print(f"🔍 Missing dependencies detected: {', '.join(missing_deps)}")
        print(f"💡 Install them with: pip install {' '.join(missing_deps)}")
    else:
        print("💡 All common dependencies seem to be installed. The issue might be with s3tokenizer or resemble-perth")
        print("💡 Try installing missing packages:")
        print("   pip install s3tokenizer resemble-perth")
        print("   Or continue with available packages for testing")
    
    # For testing, we'll create a dummy class
    print("⚠️  Creating dummy ChatterboxTTS for testing - this won't actually work!")
    
    class DummyChatterboxTTS:
        def __init__(self):
            self.sr = 24000
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        @classmethod
        def from_pretrained(cls, device):
            return cls()
        
        def generate(self, text, **kwargs):
            # Generate 1 second of dummy audio
            duration = 1.0
            sample_rate = self.sr
            samples = int(duration * sample_rate)
            # Generate a simple sine wave
            t = torch.linspace(0, duration, samples)
            freq = 440  # A4 note
            audio = 0.1 * torch.sin(2 * torch.pi * freq * t)
            return audio.unsqueeze(0)
    
    ChatterboxTTS = DummyChatterboxTTS
    print("⚠️  Using dummy TTS - install proper dependencies for real functionality")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
tts_model: Optional[ChatterboxTTS] = None

class TTSRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd", "chatterbox"] = "chatterbox"
    input: str = Field(..., max_length=4096, description="The text to generate audio for")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer", "default"] = "default"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speed of the generated audio")

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "chatterbox"
    permission: list = []
    root: str = ""
    parent: str = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatterbox",
                "object": "model",
                "created": 1677610602,
                "owned_by": "resemble-ai",
                "permission": [],
                "root": "chatterbox",
                "parent": None
            }
        }

class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

# FastAPI app
app = FastAPI(
    title="Chatterbox TTS OpenAI-Compatible API (Compatible Version)",
    description="OpenAI-compatible Text-to-Speech API using Chatterbox TTS",
    version="1.0.0"
)

def load_model(device: str = "auto") -> ChatterboxTTS:
    """Load the Chatterbox TTS model"""
    global tts_model
    
    if tts_model is not None:
        return tts_model
    
    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("🍎 Using Apple Silicon MPS")
        else:
            device = "cpu"
            logger.info("💻 Using CPU")
    
    logger.info(f"⏳ Loading Chatterbox TTS model on device: {device}")
    
    try:
        tts_model = ChatterboxTTS.from_pretrained(device)
        logger.info("✅ Model loaded successfully!")
        
        # Log VRAM usage if on GPU
        if device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"📊 VRAM usage: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
        return tts_model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def audio_to_format(audio_tensor: torch.Tensor, sample_rate: int, format: str) -> bytes:
    """Convert audio tensor to specified format"""
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
        
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    buffer = io.BytesIO()
    
    format_map = {
        "wav": "wav", "mp3": "mp3", "opus": "ogg",
        "aac": "mp4", "flac": "flac", "pcm": "wav"
    }
    
    container_format = format_map.get(format, "wav")
    
    try:
        torchaudio.save(
            buffer, audio_tensor, sample_rate,
            format=container_format,
            encoding="PCM_S" if format == "pcm" else None,
            bits_per_sample=16 if format == "pcm" else None
        )
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Failed to convert to {format}, falling back to wav: {e}")
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
        return buffer.getvalue()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        logger.info("🚀 Starting Chatterbox TTS API...")
        load_model()
    except Exception as e:
        logger.error(f"❌ Failed to load model during startup: {e}")
        logger.info("⏳ Model will be loaded on first request")

@app.get("/")
async def root():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "vram_cached_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2)
        }
    
    return {
        "status": "Chatterbox TTS API is running", 
        "version": "1.0.0 (Compatible)",
        "model_loaded": tts_model is not None,
        **gpu_info
    }

@app.get("/health")
async def health():
    """Health check endpoint with system info"""
    global tts_model
    model_loaded = tts_model is not None
    
    status = {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "device": str(tts_model.device) if model_loaded else "unknown",
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__
    }
    
    if torch.cuda.is_available():
        status.update({
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        })
    
    return status

@app.get("/voices")
async def list_voices():
    """List available voice options and cloning capabilities"""
    return {
        "built_in_voices": {
            "default": {
                "id": "default",
                "name": "Chatterbox Default",
                "description": "Built-in neural voice",
                "language": "en-US",
                "gender": "neutral",
                "usage": "Use 'default' in the voice parameter"
            }
        },
        "openai_compatible_voices": {
            "note": "These names are accepted for OpenAI compatibility but all use the same default voice",
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        },
        "voice_cloning": {
            "enabled": True,
            "description": "Clone any voice by uploading a reference audio file",
            "endpoint": "/v1/audio/speech/custom",
            "supported_formats": ["wav", "mp3", "flac", "opus", "aac"],
            "recommended_duration": "3-30 seconds",
            "parameters": {
                "exaggeration": {
                    "description": "Emotion/expression level",
                    "range": "0.0-2.0",
                    "default": 0.5,
                    "examples": {
                        "neutral": 0.3,
                        "normal": 0.5, 
                        "expressive": 0.8,
                        "dramatic": 1.5
                    }
                },
                "temperature": {
                    "description": "Sampling randomness/creativity",
                    "range": "0.1-1.0", 
                    "default": 0.8,
                    "examples": {
                        "consistent": 0.6,
                        "balanced": 0.8,
                        "creative": 0.95
                    }
                },
                "cfg_weight": {
                    "description": "Classifier-free guidance strength",
                    "range": "0.0-1.0",
                    "default": 0.5,
                    "note": "Higher values = more faithful to reference"
                }
            }
        },
        "tips": {
            "best_results": [
                "Use clear, high-quality reference audio",
                "3-10 seconds of clean speech works best",
                "Avoid background noise, music, or distortion",
                "Natural conversational speech is better than reading",
                "Multiple short clips can work better than one long clip"
            ],
            "exaggeration_guide": [
                "0.2-0.4: Very neutral, professional",
                "0.5-0.7: Natural conversation", 
                "0.8-1.2: Expressive, engaging",
                "1.3-2.0: Dramatic, theatrical"
            ]
        },
        "examples": {
            "standard_tts": {
                "endpoint": "/v1/audio/speech",
                "example": {
                    "model": "chatterbox",
                    "input": "Hello world!",
                    "voice": "default",
                    "response_format": "mp3"
                }
            },
            "voice_cloning": {
                "endpoint": "/v1/audio/speech/custom",
                "method": "POST (multipart/form-data)",
                "fields": {
                    "text": "Text to synthesize",
                    "voice_file": "Reference audio file",
                    "response_format": "wav",
                    "exaggeration": "0.7",
                    "temperature": "0.8",
                    "cfg_weight": "0.5"
                }
            }
        }
    }
async def list_models():
    """List available models (OpenAI compatible)"""
    current_time = int(time.time())
    models = [
        ModelInfo(
            id="chatterbox",
            created=current_time,
            owned_by="resemble-ai",
            root="chatterbox"
        ),
        ModelInfo(
            id="tts-1", 
            created=current_time,
            owned_by="chatterbox-api",
            root="tts-1"
        ),
        ModelInfo(
            id="tts-1-hd",
            created=current_time, 
            owned_by="chatterbox-api",
            root="tts-1-hd"
        )
    ]
    return ModelsResponse(data=models)

@app.get("/models")
async def list_models_detailed():
    """List available models with detailed information"""
    global tts_model
    current_time = int(time.time())
    
    model_status = "loaded" if tts_model is not None else "not_loaded"
    device_info = str(tts_model.device) if tts_model is not None else "unknown"
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "cuda_version": torch.version.cuda
        }
    else:
        gpu_info = {"gpu_available": False}
    
    models = {
        "chatterbox": {
            "id": "chatterbox",
            "name": "Chatterbox TTS",
            "description": "High-quality neural text-to-speech with voice cloning capabilities",
            "type": "text-to-speech",
            "created": current_time,
            "owned_by": "resemble-ai",
            "status": model_status,
            "device": device_info,
            "features": [
                "voice_cloning",
                "emotion_control", 
                "multi_language",
                "real_time_inference"
            ],
            "supported_formats": ["wav", "mp3", "flac", "opus", "aac", "pcm"],
            "supported_voices": ["default", "alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "sample_rate": 24000,
            "max_text_length": 4096
        },
        "tts-1": {
            "id": "tts-1", 
            "name": "TTS-1 (Chatterbox Backend)",
            "description": "OpenAI TTS-1 compatible endpoint using Chatterbox TTS",
            "type": "text-to-speech",
            "created": current_time,
            "owned_by": "chatterbox-api",
            "status": model_status,
            "device": device_info,
            "features": ["openai_compatible"],
            "backend": "chatterbox"
        },
        "tts-1-hd": {
            "id": "tts-1-hd",
            "name": "TTS-1-HD (Chatterbox Backend)", 
            "description": "OpenAI TTS-1-HD compatible endpoint using Chatterbox TTS",
            "type": "text-to-speech",
            "created": current_time,
            "owned_by": "chatterbox-api", 
            "status": model_status,
            "device": device_info,
            "features": ["openai_compatible", "high_quality"],
            "backend": "chatterbox"
        }
    }
    
    return {
        "models": models,
        "system_info": {
            "torch_version": torch.__version__,
            "torchaudio_version": torchaudio.__version__,
            **gpu_info
        },
        "api_info": {
            "version": "1.0.0 (Compatible)",
            "endpoints": [
                "/v1/audio/speech",
                "/v1/audio/speech/custom", 
                "/v1/models",
                "/models",
                "/health"
            ]
        }
    }

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """Generate speech from text"""
    global tts_model
    
    if tts_model is None:
        try:
            logger.info("⏳ Loading model for first request...")
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load TTS model: {str(e)}")
    
    try:
        logger.info(f"🎤 Generating speech for: '{request.input[:50]}{'...' if len(request.input) > 50 else ''}' (voice: {request.voice})")
        
        start_time = time.time()
        
        # Generate audio
        if hasattr(tts_model, 'generate'):
            audio_tensor = tts_model.generate(
                text=request.input,
                temperature=0.8,
                exaggeration=0.5
            )
        else:
            # Dummy generation for testing
            audio_tensor = tts_model.generate(request.input)
        
        generation_time = time.time() - start_time
        
        # Convert to requested format
        audio_bytes = audio_to_format(audio_tensor, tts_model.sr, request.response_format)
        
        content_types = {
            "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", 
            "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm"
        }
        
        content_type = content_types.get(request.response_format, "audio/wav")
        
        # Log performance stats
        audio_duration = len(audio_tensor[0]) / tts_model.sr
        rtf = generation_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"✅ Generated {len(audio_bytes)} bytes in {generation_time:.2f}s (RTF: {rtf:.2f}x, {audio_duration:.1f}s audio)")
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Generation-Time": str(generation_time),
                "X-Audio-Duration": str(audio_duration),
                "X-RTF": str(rtf)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chatterbox TTS OpenAI-Compatible API Server (Compatible Version)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Compatible Chatterbox TTS API Server")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
    print(f"🌐 Server will be available at: http://{args.host}:{args.port}")
    print(f"📖 API documentation: http://{args.host}:{args.port}/docs")
    
    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
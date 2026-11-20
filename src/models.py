"""Model loading and shared runtime configuration."""

import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad


# STT
def load_whisper_model(
    model_name: str = "large-v3-turbo",
    device: str = "mps",
    compute_type: str = "float16",
) -> WhisperModel:
    """Load and configure a faster-whisper speech recognition model.

    Args:
        model_name: Hugging Face or faster-whisper model identifier.
        device: Inference device, such as ``mps`` or ``cpu``.
        compute_type: Whisper compute precision.
    """
    print("Loading faster-whisper model...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print("Model loaded.")
    return model


# VAD
def load_vad(
    target_rate: int,
    silence_duration_ms: int,
    threshold: float = 0.75,
) -> VADIterator:
    """Load Silero voice activity detection with sentence timing defaults.

    Args:
        target_rate: Audio sample rate consumed by the VAD.
        silence_duration_ms: Silence duration that ends an active utterance.
        threshold: Speech probability required to enter voice activity state.
    """
    print("Loading Silero VAD...")
    torch.set_num_threads(1)
    vad = VADIterator(
        load_silero_vad(),
        threshold=threshold,
        sampling_rate=target_rate,
        min_silence_duration_ms=silence_duration_ms,
        speech_pad_ms=300,
    )
    print("VAD loaded.")
    return vad


# TTS
import os

# Disable huggingface loading as we already have it downloaded
os.environ["HF_HUB_OFFLINE"] = "1"

import mlx.core as mx
from mlx_audio.tts.utils import load_model


def load_tts_model(model_name: str = "mlx-community/OmniVoice-bf16"):
    """Load an MLX text-to-speech model for the process lifetime.

    Args:
        model_name: Hugging Face model identifier available to MLX Audio.
    """
    print("Loading TTS model...")
    model = load_model(model_name)  # ty: ignore[invalid-argument-type]
    print("TTS model loaded.")
    return model


mx.random.seed(42)

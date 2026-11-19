import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad


def load_whisper_model(
    model_name: str = "large-v3-turbo",
    device: str = "mps",
    compute_type: str = "float16",
) -> WhisperModel:
    """Load the speech recognition model once for the process lifetime."""
    print("Loading faster-whisper model...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print("Model loaded.")
    return model


def load_vad(
    target_rate: int,
    silence_duration_ms: int,
) -> VADIterator:
    """Load Silero VAD once for the process lifetime."""
    print("Loading Silero VAD...")
    torch.set_num_threads(1)
    vad = VADIterator(
        load_silero_vad(),
        sampling_rate=target_rate,
        min_silence_duration_ms=silence_duration_ms,
        speech_pad_ms=300,
    )
    print("VAD loaded.")
    return vad

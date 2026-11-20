import numpy as np
import sounddevice as sd
import torch

BLACKHOLE_DEVICE = "BlackHole 2ch"


def to_sounddevice_audio(audio):
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().cpu().numpy()
    else:
        arr = np.asarray(audio)

    arr = arr.astype(np.float32)

    # Convert (channels, samples) -> (samples, channels) if 2D
    if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    # If mono (1D or (N, 1)) and BlackHole 2ch expects 2 channels, duplicate to stereo
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)

    return np.clip(arr, -1.0, 1.0)

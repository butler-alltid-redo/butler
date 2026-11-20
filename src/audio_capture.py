"""VAD-based capture and resampling helpers for microphone audio."""

import queue
import threading
from collections.abc import Callable

import numpy as np
import torch
from scipy.signal import resample_poly


def capture_audio(
    stream,
    vad,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    *,
    source_rate: int,
    target_rate: int,
    channels: int,
    frames_per_read: int,
    pre_roll_chunks: int,
    max_utterance_seconds: int,
    speech_confirm_ms: int = 350,
    on_speech_start: Callable[[], None] | None = None,
) -> None:
    """Capture speech until silence or the maximum utterance length is reached.

    Args:
        stream: A PyAudio input stream.
        vad: A Silero VAD iterator that expects 16 kHz mono float samples.
        audio_queue: Destination for completed utterances and a final ``None``.
        stop_event: Signals the capture thread to stop.
        source_rate: Sample rate produced by the PyAudio stream.
        target_rate: Required VAD and Whisper sample rate.
        channels: Number of channels in the PyAudio stream.
        frames_per_read: Number of source frames read per loop iteration.
        pre_roll_chunks: Number of recent chunks retained before speech begins.
        max_utterance_seconds: Maximum duration of one captured utterance.
        speech_confirm_ms: Sustained speech required before a start callback fires.
        on_speech_start: Optional callback invoked immediately when VAD detects speech.
    """
    pre_roll = []
    recording = []
    is_recording = False
    is_confirming = False
    confirmation_chunks = []
    speech_confirm_chunks = max(
        1,
        int(speech_confirm_ms * target_rate / 1000 / 512),
    )
    max_chunks = int(max_utterance_seconds * target_rate / 512)

    try:
        while not stop_event.is_set():
            data = stream.read(frames_per_read, exception_on_overflow=False)

            stereo = np.frombuffer(data, dtype=np.float32).reshape(-1, channels)
            mono = stereo.mean(axis=1)
            audio = resample_poly(mono, up=1, down=source_rate // target_rate).astype(
                np.float32
            )

            with torch.inference_mode():
                event = vad(torch.from_numpy(audio))

            if not is_recording and not is_confirming:
                pre_roll.append(audio.copy())
                pre_roll = pre_roll[-pre_roll_chunks:]

                if event and "start" in event:
                    recording = pre_roll.copy()
                    pre_roll.clear()
                    is_confirming = True
                    confirmation_chunks = [audio.copy()]
            elif is_confirming:
                confirmation_chunks.append(audio.copy())

                if len(confirmation_chunks) >= speech_confirm_chunks:
                    print("\nSpeech detected...")
                    if on_speech_start is not None:
                        on_speech_start()
                    recording.extend(confirmation_chunks)
                    confirmation_chunks = []
                    is_confirming = False
                    is_recording = True
                elif event and "end" in event:
                    recording = []
                    confirmation_chunks = []
                    is_confirming = False
                    vad.reset_states()
            else:
                recording.append(audio.copy())

                if (event and "end" in event) or len(recording) >= max_chunks:
                    print("Speech finished.")
                    audio_queue.put(np.concatenate(recording))
                    recording = []
                    pre_roll.clear()
                    is_recording = False
                    vad.reset_states()

    except Exception as error:  # noqa: BLE001 - audio backends vary and must not crash silently
        if not stop_event.is_set():
            print(f"Audio capture error: {error}")
    finally:
        audio_queue.put(None)

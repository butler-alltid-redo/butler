import queue
import threading

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
) -> None:
    """Capture audio and queue completed speech utterances."""
    pre_roll = []
    recording = []
    is_recording = False
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

            if not is_recording:
                pre_roll.append(audio.copy())
                pre_roll = pre_roll[-pre_roll_chunks:]

                if event and "start" in event:
                    print("\nSpeech detected...")
                    recording = pre_roll.copy()
                    pre_roll.clear()
                    is_recording = True
            else:
                recording.append(audio.copy())

                if (event and "end" in event) or len(recording) >= max_chunks:
                    print("Speech finished.")
                    audio_queue.put(np.concatenate(recording))
                    recording = []
                    pre_roll.clear()
                    is_recording = False
                    vad.reset_states()

    except Exception as error:
        if not stop_event.is_set():
            print(f"Audio capture error: {error}")
    finally:
        audio_queue.put(None)

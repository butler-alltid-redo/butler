import queue
import re
import threading

import numpy as np
import pyaudio
import pydantic_ai

from play_sound import to_sounddevice_audio

pydantic_ai.BANNER_ENABLED = False  # ty: ignore[invalid-assignment]

import sounddevice as sd
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

from audio_capture import capture_audio
from models import load_tts_model, load_vad, load_whisper_model

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma4:e4b"
AGENT_INSTRUCTIONS = """
Du är en naturlig, kunnig och avslappnad svensk röstassistent. Svara kortfattat, vänligt och konversationsmässigt. Anpassa språket för tal, inte text. Undvik onödiga artighetsfraser, upprepningar och långa utläggningar. Använd aldrig emojis, markdown eller listformatering.

Använd gärna följande icke-verbala ljudtaggar när de passar naturligt i samtalet:
[laughter], [sigh], [confirmation-en], [question-en], [question-ah], [question-oh], [question-ei], [question-yi], [surprise-ah], [surprise-oh], [surprise-wa], [surprise-yo], [dissatisfaction-hnn].

Använd taggarna sparsamt och bara när de tillför något. Tvinga aldrig fram känslouttryck. Var personlig, spontan och mänsklig, inte robotlik. Var ärlig om du inte vet något och ställ följdfrågor när det behövs.
"""

# STT
STT_INPUT_DEVICE = "BlackHole 16ch"
SOURCE_RATE, TARGET_RATE = 48000, 16000
CHANNELS, FRAMES_PER_READ = 2, 1536
SILENCE_MS, PRE_ROLL, MAX_UTTERANCE = 1000, 10, 120

# LLM
KEEP_ALIVE = "240h"

# TTS
TTS_OUTPUT_DEVICE = "BlackHole 2ch"
CHUNK_PUNCTUATION = re.compile(r"[.!?]")
tts_queue: queue.Queue[str | None] = queue.Queue()
playback_queue: queue.Queue[np.ndarray | None] = queue.Queue()
playback_done = threading.Event()
playback_stream: sd.OutputStream | None = None
playback_audio: np.ndarray | None = None

# Initialize models and clients
whisper = load_whisper_model()
vad = load_vad(TARGET_RATE, SILENCE_MS)
tts_model = load_tts_model()

# Use Ollama's OpenAI-compatible API while letting Pydantic AI retain the
# complete message history between transcribed utterances.
ollama_provider = OpenAICompatibleProvider(
    "ollama",
    base_url=f"{OLLAMA_URL}/v1",
    api_key="ollama",
)


def split_stream_chunks(text: str, pending: str = "") -> tuple[list[str], str]:
    """Split streamed text at periods, question marks, and exclamation marks."""
    pending += text
    chunks: list[str] = []

    while match := CHUNK_PUNCTUATION.search(pending):
        chunk = pending[: match.end()].strip()
        pending = pending[match.end() :].lstrip()
        if chunk:
            chunks.append(chunk)

    return chunks, pending


def speak(text: str) -> None:
    tts_queue.put(text)


def playback_callback(
    outdata: np.ndarray,
    frames: int,
    _time,
    _status,
) -> None:
    global playback_audio

    outdata.fill(0)
    written = 0

    while written < frames:
        if playback_audio is None:
            try:
                playback_audio = playback_queue.get_nowait()
            except queue.Empty:
                break

            if playback_audio is None:
                playback_done.set()
                playback_audio = None
                break

        remaining = frames - written
        available = len(playback_audio) - 0
        count = min(remaining, available)
        outdata[written : written + count] = playback_audio[:count]

        if count == len(playback_audio):
            playback_audio = None
        else:
            playback_audio = playback_audio[count:]

        written += count


def tts_worker() -> None:
    global playback_stream

    while (text := tts_queue.get()) is not None:
        for result in tts_model.generate(
            text=text,
            ref_audio="gago.wav",
            ref_text="Jag kommer aldrig att glömma hur det ojämna jordgolvet skavde mot mina bara knän.",
        ):
            samples = to_sounddevice_audio(result.audio)
            if playback_stream is None:
                playback_stream = sd.OutputStream(
                    samplerate=result.sample_rate,
                    channels=2,
                    dtype=np.float32,
                    device=TTS_OUTPUT_DEVICE,
                    callback=playback_callback,
                )
                playback_stream.start()
            playback_queue.put(samples)
        tts_queue.task_done()

    playback_queue.put(None)
    if playback_stream is None:
        playback_done.set()


def agent():
    agent = Agent(
        OpenAIChatModel(OLLAMA_MODEL, provider=ollama_provider),
        instructions=AGENT_INSTRUCTIONS,
        model_settings={"extra_body": {"keep_alive": KEEP_ALIVE}},
    )
    conversation_history: list[ModelMessage] | None = None

    # Initialize audio
    audio = pyaudio.PyAudio()
    input_device_index = next(
        (
            index
            for index in range(audio.get_device_count())
            if audio.get_device_info_by_index(index)["name"] == STT_INPUT_DEVICE
            and audio.get_device_info_by_index(index)["maxInputChannels"] >= CHANNELS
        ),
        None,
    )
    if input_device_index is None:
        audio.terminate()
        raise RuntimeError(
            f"Could not find a {CHANNELS}-channel input device named "
            f"'{STT_INPUT_DEVICE}'."
        )

    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=SOURCE_RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=FRAMES_PER_READ,
    )

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_audio,
        args=(stream, vad, audio_queue, stop_event),
        kwargs={
            "source_rate": SOURCE_RATE,
            "target_rate": TARGET_RATE,
            "channels": CHANNELS,
            "frames_per_read": FRAMES_PER_READ,
            "pre_roll_chunks": PRE_ROLL,
            "max_utterance_seconds": MAX_UTTERANCE,
        },
        daemon=True,
    )

    capture_thread.start()
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    try:
        print(f"Listening via {STT_INPUT_DEVICE} (device {input_device_index})...")

        while (audio_data := audio_queue.get()) is not None:
            print(f"\nRecording: {len(audio_data) / TARGET_RATE:.1f}s")

            segments, _ = whisper.transcribe(
                audio_data,
                beam_size=5,
                language="sv",
                vad_filter=False,
                condition_on_previous_text=False,
            )

            transcript = " ".join(s.text.strip() for s in segments if s.text.strip())
            if not transcript:
                continue

            print(f"You: {transcript}")

            try:
                with agent.run_stream_sync(
                    transcript,
                    message_history=conversation_history,
                ) as result:
                    pending = ""
                    print("LLM: ", end="", flush=True)

                    for delta in result.stream_text(delta=True, debounce_by=None):
                        print(delta, end="", flush=True)
                        chunks, pending = split_stream_chunks(delta, pending)
                        for chunk in chunks:
                            speak(chunk)

                    if pending.strip():
                        speak(pending.strip())

                    print()
                    conversation_history = result.all_messages()

            except RuntimeError as e:
                print(f"Assistant error: {e}")

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()
        capture_thread.join(timeout=3)
        tts_queue.put(None)
        tts_thread.join()
        playback_done.wait(timeout=30)

        if playback_stream is not None:
            playback_stream.stop()
            playback_stream.close()

        if stream.is_active():
            stream.stop_stream()

        stream.close()
        audio.terminate()
        print("Stopped.")

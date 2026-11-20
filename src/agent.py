"""Real-time voice assistant orchestration.

This module connects speech capture, speech-to-text, a streaming Pydantic AI
agent, and text-to-speech.  The model stream is divided into sentence-sized
chunks, while a separate worker generates and queues audio so LLM processing
can continue without waiting for playback.
"""

import queue
import re
import threading
from dataclasses import dataclass, field

import numpy as np
import pyaudio
import pydantic_ai

from play_sound import to_sounddevice_audio

pydantic_ai.BANNER_ENABLED = False  # ty: ignore[invalid-assignment]

import sounddevice as sd
from pydantic_ai import Agent, CancellationToken
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

from audio_capture import capture_audio
from models import load_tts_model, load_vad, load_whisper_model

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma4:e4b"
AGENT_INSTRUCTIONS = """
Du heter Butler. Din ägare heter Sigge. Du är en naturlig, kunnig och avslappnad svensk röstassistent. Svara kortfattat, vänligt och konversationsmässigt. Anpassa språket för tal, inte text. Undvik onödiga artighetsfraser, upprepningar och långa utläggningar. Använd aldrig emojis, markdown eller listformatering.

Använd gärna följande icke-verbala ljudtaggar när de passar naturligt i samtalet:
[laughter], [sigh], [confirmation-en], [question-en], [question-ah], [question-oh], [question-ei], [question-yi], [surprise-ah], [surprise-oh], [surprise-wa], [surprise-yo], [dissatisfaction-hnn].

Använd taggarna sparsamt och bara när de tillför något. Tvinga aldrig fram känslouttryck. Var personlig, spontan och mänsklig, inte robotlik. Var ärlig om du inte vet något och ställ följdfrågor när det behövs.
"""

# STT
STT_INPUT_DEVICE = "BlackHole 16ch"
SOURCE_RATE, TARGET_RATE = 48000, 16000
CHANNELS, FRAMES_PER_READ = 2, 1536
SILENCE_MS, PRE_ROLL, MAX_UTTERANCE = 1000, 10, 120
VAD_THRESHOLD = 0.75
SPEECH_CONFIRM_MS = 350

# LLM
KEEP_ALIVE = "240h"

# TTS
TTS_OUTPUT_DEVICE = "BlackHole 2ch"
CHUNK_PUNCTUATION = re.compile(r"[.!?]+")


@dataclass
class AgentSession:
    """Per-call state shared by the capture, LLM, and TTS workers."""

    stop_event: threading.Event = field(default_factory=threading.Event)
    audio_queue: queue.Queue[np.ndarray | None] = field(default_factory=queue.Queue)
    tts_queue: queue.Queue[str | None] = field(default_factory=queue.Queue)
    playback_queue: queue.Queue[np.ndarray | None] = field(default_factory=queue.Queue)
    playback_done: threading.Event = field(default_factory=threading.Event)
    audio: pyaudio.PyAudio | None = None
    input_stream: pyaudio.Stream | None = None
    capture_thread: threading.Thread | None = None
    tts_thread: threading.Thread | None = None
    playback_stream: sd.OutputStream | None = None
    playback_audio: np.ndarray | None = None
    playback_lock: threading.Lock = field(default_factory=threading.Lock)
    interruption_event: threading.Event = field(default_factory=threading.Event)
    active_cancellation_lock: threading.Lock = field(default_factory=threading.Lock)
    active_cancellation_token: CancellationToken | None = None
    tts_generation: int = 0

# Initialize models and clients
whisper = load_whisper_model()
vad = load_vad(TARGET_RATE, SILENCE_MS, threshold=VAD_THRESHOLD)
tts_model = load_tts_model()

# Use Ollama's OpenAI-compatible API while letting Pydantic AI retain the
# complete message history between transcribed utterances.
ollama_provider = OpenAICompatibleProvider(
    "ollama",
    base_url=f"{OLLAMA_URL}/v1",
    api_key="ollama",
)


def split_stream_chunks(text: str, pending: str = "") -> tuple[list[str], str]:
    """Split delta text into punctuation-delimited speech chunks.

    Adjacent punctuation marks such as ``...``, ``?!``, and ``!!`` are kept
    together as one boundary instead of creating a separate chunk for each mark.

    Args:
        text: A newly received LLM text delta.
        pending: Text waiting for a period, question mark, or exclamation mark.

    Returns:
        A pair containing complete chunks and the remaining unterminated text.
    """
    pending += text
    chunks: list[str] = []

    while match := CHUNK_PUNCTUATION.search(pending):
        chunk = pending[: match.end()].strip()
        pending = pending[match.end() :].lstrip()
        if chunk:
            chunks.append(chunk)

    return chunks, pending


def speak(text: str, session: AgentSession) -> None:
    """Queue text for synthesis without blocking the LLM stream."""
    if not session.interruption_event.is_set():
        session.tts_queue.put(text)


def clear_queue(item_queue: queue.Queue) -> None:
    """Remove queued work without blocking the worker or audio callback."""
    with item_queue.mutex:
        item_queue.queue.clear()


def interrupt_assistant(session: AgentSession) -> None:
    """Cancel active LLM generation and discard all pending TTS/playback."""
    session.interruption_event.set()
    session.tts_generation += 1
    clear_queue(session.tts_queue)
    clear_queue(session.playback_queue)

    with session.playback_lock:
        session.playback_audio = None

    with session.active_cancellation_lock:
        token = session.active_cancellation_token

    if token is not None and not token.cancelled:
        token.cancel()


def playback_callback(
    session: AgentSession,
    outdata: np.ndarray,
    frames: int,
    _time,
    _status,
) -> None:
    """Drain queued stereo audio into one PortAudio callback buffer.

    The callback silently fills any remaining frames when synthesis or queued
    audio is temporarily unavailable.  It never blocks the audio thread.
    """
    with session.playback_lock:
        outdata.fill(0)
        written = 0

        while written < frames:
            if session.playback_audio is None:
                try:
                    session.playback_audio = session.playback_queue.get_nowait()
                except queue.Empty:
                    break

                if session.playback_audio is None:
                    session.playback_done.set()
                    session.playback_audio = None
                    break

            remaining = frames - written
            available = len(session.playback_audio)
            count = min(remaining, available)
            outdata[written : written + count] = session.playback_audio[:count]

            if count == len(session.playback_audio):
                session.playback_audio = None
            else:
                session.playback_audio = session.playback_audio[count:]

            written += count


def tts_worker(session: AgentSession) -> None:
    """Synthesize queued text and append its audio to the FIFO playback queue."""
    while (text := session.tts_queue.get()) is not None:
        generation = session.tts_generation
        for result in tts_model.generate(
            text=text,
            ref_audio="gago.wav",
            ref_text="Jag kommer aldrig att glömma hur det ojämna jordgolvet skavde mot mina bara knän.",
        ):
            if generation != session.tts_generation:
                break
            samples = to_sounddevice_audio(result.audio)
            if session.playback_stream is None:
                session.playback_stream = sd.OutputStream(
                    samplerate=result.sample_rate,
                    channels=2,
                    dtype=np.float32,
                    device=TTS_OUTPUT_DEVICE,
                    callback=lambda *args, _session=session: playback_callback(
                        _session, *args
                    ),
                )
                session.playback_stream.start()
            session.playback_queue.put(samples)
        session.tts_queue.task_done()

    session.playback_queue.put(None)
    if session.playback_stream is None:
        session.playback_done.set()


def transcribe_utterance(audio_data: np.ndarray) -> str:
    """Transcribe one utterance and recover from common transient Whisper failures.

    Args:
        audio_data: Resampled mono float32 audio at ``TARGET_RATE``.

    Returns:
        Transcribed Swedish text, or an empty string for unusable or failed audio.
    """
    if (
        audio_data is None
        or len(audio_data) < TARGET_RATE * 0.25
        or not np.isfinite(audio_data).all()
    ):
        return ""

    audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

    for beam_size in (5, 1):
        try:
            segments, _ = whisper.transcribe(
                audio_data,
                beam_size=beam_size,
                language="sv",
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )
        except (IndexError, RuntimeError, ValueError) as error:
            if beam_size == 5:
                print(f"Whisper retry after transcription error: {error}")
                continue
            print(f"Whisper transcription failed: {error}")

    return ""


def stream_assistant_response(
    assistant: Agent,
    prompt: str,
    conversation_history: list[ModelMessage] | None,
    output_label: str,
    session: AgentSession,
) -> list[ModelMessage] | None:
    """Stream one agent response to text, TTS chunks, and message history.

    Args:
        assistant: Pydantic AI agent used for the response.
        prompt: User prompt or generated session prompt.
        conversation_history: Prior Pydantic AI messages, if any.
        output_label: Console label printed before streamed text.

    Returns:
        Updated message history unless the response was interrupted.
    """
    session.interruption_event.clear()
    cancellation_token = CancellationToken()

    with session.active_cancellation_lock:
        session.active_cancellation_token = cancellation_token

    try:
        with assistant.run_stream_sync(
            prompt,
            message_history=conversation_history,
            cancellation_token=cancellation_token,
        ) as result:
            pending = ""
            print(f"{output_label}: ", end="", flush=True)

            for delta in result.stream_text(delta=True, debounce_by=None):
                if session.interruption_event.is_set() or session.stop_event.is_set():
                    if session.stop_event.is_set():
                        cancellation_token.cancel()
                    break

                print(delta, end="", flush=True)
                chunks, pending = split_stream_chunks(delta, pending)

                for chunk in chunks:
                    if session.interruption_event.is_set() or session.stop_event.is_set():
                        break
                    speak(chunk, session)

            if (
                pending.strip()
                and not session.interruption_event.is_set()
                and not session.stop_event.is_set()
            ):
                speak(pending.strip(), session)

            print()
            if not session.interruption_event.is_set() and not session.stop_event.is_set():
                return result.all_messages()

    except RunCancelled:
        print(f"\n{output_label} cancelled.")
    except RuntimeError as error:
        print(f"{output_label} error: {error}")
    finally:
        with session.active_cancellation_lock:
            if session.active_cancellation_token is cancellation_token:
                session.active_cancellation_token = None

    return conversation_history


def agent(session: AgentSession | None = None):
    """Run the interactive voice-assistant loop until interrupted.

    Captures utterances, transcribes them, streams LLM responses, and queues
    punctuation-delimited text for TTS.  The loop retains Pydantic AI message
    history across utterances.

    Args:
        session: Per-call state; a new session is created when omitted.

    Raises:
        RuntimeError: If the configured BlackHole input device is unavailable.
    """
    if session is None:
        session = AgentSession()
    stop_event = session.stop_event

    agent = Agent(
        OpenAIChatModel(OLLAMA_MODEL, provider=ollama_provider),
        instructions=AGENT_INSTRUCTIONS,
        model_settings={"extra_body": {"keep_alive": KEEP_ALIVE}},
    )
    conversation_history: list[ModelMessage] | None = None

    greeting_prompt = (
        "Säg ett kort, unikt och avslappnat hälsningsfras på svenska "
        "som om du precis har svarat i telefon. Svara bara med själva hälsningen, "
        "inte citattecken eller förklaringar. Du kan nämna ägarens namn Sigge om du vill. "
        "Exempelvis 'Tjeeenare Sigge!', 'Butler speaking.', 'Hej Sigge, vad kan jag hjälpa dig med?', eller 'Hallå där!'."
    )

    # Initialize audio
    session.audio = pyaudio.PyAudio()
    audio = session.audio
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

    session.input_stream = audio.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=SOURCE_RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=FRAMES_PER_READ,
    )
    stream = session.input_stream
    audio_queue = session.audio_queue

    session.capture_thread = threading.Thread(
        target=capture_audio,
        args=(stream, vad, audio_queue, stop_event),
        kwargs={
            "source_rate": SOURCE_RATE,
            "target_rate": TARGET_RATE,
            "channels": CHANNELS,
            "frames_per_read": FRAMES_PER_READ,
            "pre_roll_chunks": PRE_ROLL,
            "max_utterance_seconds": MAX_UTTERANCE,
            "speech_confirm_ms": SPEECH_CONFIRM_MS,
            "on_speech_start": lambda: interrupt_assistant(session),
        },
        daemon=True,
    )

    session.capture_thread.start()
    session.tts_thread = threading.Thread(
        target=tts_worker,
        args=(session,),
        daemon=True,
    )
    session.tts_thread.start()

    try:
        print(f"Listening via {STT_INPUT_DEVICE} (device {input_device_index})...")

        conversation_history = stream_assistant_response(
            agent,
            greeting_prompt,
            conversation_history,
            "Greeting",
            session,
        )

        while not stop_event.is_set():
            try:
                audio_data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio_data is None:
                break
            print(f"\nRecording: {len(audio_data) / TARGET_RATE:.1f}s")
            transcript = transcribe_utterance(audio_data)
            if not transcript:
                continue

            print(f"You: {transcript}")

            conversation_history = stream_assistant_response(
                agent,
                transcript,
                conversation_history,
                "LLM",
                session,
            )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()
        if session.capture_thread is not None:
            session.capture_thread.join(timeout=3)
        session.tts_queue.put(None)
        if session.tts_thread is not None:
            session.tts_thread.join()
        session.playback_done.wait(timeout=30)

        if session.playback_stream is not None:
            session.playback_stream.stop()
            session.playback_stream.close()
            session.playback_stream = None

        if session.input_stream is not None:
            if session.input_stream.is_active():
                session.input_stream.stop_stream()
            session.input_stream.close()
            session.input_stream = None

        if session.audio is not None:
            session.audio.terminate()
            session.audio = None
        print("Stopped.")

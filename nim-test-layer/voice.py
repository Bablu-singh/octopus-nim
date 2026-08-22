"""Listening and speaking, locally and for free.

The octopus could always be typed at. This gives it ears and a voice: send a voice note
from WhatsApp or Discord and it becomes a task; get the answer back as audio you can play
while doing something else. That is the difference between controlling this from a phone
and merely being able to.

Both halves run on this machine, which matters as much here as it does for the models:

  Piper for speech        63 MB ONNX, roughly 8x realtime on this CPU. It synthesises a
                          six-second sentence in under a second, so an answer is spoken
                          about as fast as it can be read.
  faster-whisper for ears base.en, int8, roughly 6x realtime. Transcribes a voice note in
                          a fraction of its length.

No API, no key, no quota, nothing sent anywhere — the same constraint the providers
follow. There is no GPU on this machine and neither of these needs one.

--- Why the models are loaded lazily ---------------------------------------

Together they are a couple of hundred megabytes of RAM, and most runs never speak. So
nothing is loaded until the first call that actually needs it, and then it is kept: a
model reloaded per utterance would spend more time loading than speaking.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import threading
import wave
from pathlib import Path

HERE = Path(__file__).parent
VOICES = Path(os.getenv("VOICE_DIR", str(HERE / "voices")))

ENABLED = os.getenv("ENABLE_VOICE", "1") != "0"
# Any voice from rhasspy/piper-voices. 'medium' is the quality/size sweet spot; 'low' is
# half the size and noticeably flatter.
TTS_VOICE = os.getenv("VOICE_TTS_MODEL", "en_US-lessac-medium")
# base.en is the smallest model that transcribes normal speech reliably. 'small.en' is
# better on accents and about three times slower.
STT_MODEL = os.getenv("VOICE_STT_MODEL", "base.en")
# Spoken answers are capped: an agent can produce two thousand words, and nobody wants to
# listen to that on a phone. The text version is always sent as well.
MAX_SPEAK_CHARS = int(os.getenv("VOICE_MAX_CHARS", "1200"))

_tts = None
_stt = None
_tts_lock = threading.Lock()
_stt_lock = threading.Lock()
_state = {"tts_error": "", "stt_error": ""}


def available() -> tuple[bool, str]:
    if not ENABLED:
        return False, "ENABLE_VOICE is 0."
    try:
        import piper          # noqa: F401
        import faster_whisper  # noqa: F401
    except ImportError as err:
        return False, f"missing dependency: {err.name} — pip install piper-tts faster-whisper"
    return True, "ready"


# --- text going out ---------------------------------------------------------

_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_MARK = re.compile(r"[*_#>|]+")
_BULLET = re.compile(r"(?m)^\s*[-*•]\s+")


def speakable(text: str) -> str:
    """Turn an agent's Markdown into something worth hearing.

    Read literally, Markdown is unbearable: asterisks become "asterisk", a URL becomes a
    minute of alphabet, and a forty-line code block is unlistenable. Code and links are
    named rather than read, formatting is dropped, and the result is capped — the full
    text is always delivered alongside, so nothing is lost by not speaking it.
    """
    text = _FENCE.sub(" (code block omitted) ", text or "")
    text = _INLINE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" (link) ", text)
    text = _BULLET.sub("", text)
    text = _MARK.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) > MAX_SPEAK_CHARS:
        # Cut on a sentence boundary if there is one nearby, so it does not stop mid-word.
        cut = text[:MAX_SPEAK_CHARS]
        dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = (cut[:dot + 1] if dot > MAX_SPEAK_CHARS * 0.6 else cut) + " … "
        text += "That is the short version; the rest is in the text."
    return text


def _load_tts():
    global _tts
    with _tts_lock:
        if _tts is None:
            from piper import PiperVoice
            from piper.download_voices import download_voice
            VOICES.mkdir(parents=True, exist_ok=True)
            model = VOICES / f"{TTS_VOICE}.onnx"
            if not model.exists():
                print(f"[voice] downloading {TTS_VOICE} …", flush=True)
                download_voice(TTS_VOICE, VOICES)
            _tts = PiperVoice.load(model)
            print(f"[voice] tts ready: {TTS_VOICE}", flush=True)
    return _tts


def _synth(text: str) -> bytes:
    voice = _load_tts()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w)
    return buf.getvalue()


async def speak(text: str) -> bytes:
    """Text in, WAV out. Empty bytes if voice is off or synthesis fails.

    Runs in a worker thread: synthesis is CPU-bound and would otherwise stall the loop
    every agent is streaming through.
    """
    ok, why = available()
    if not ok:
        return b""
    body = speakable(text)
    if not body:
        return b""
    try:
        return await asyncio.get_running_loop().run_in_executor(None, _synth, body)
    except Exception as err:
        _state["tts_error"] = f"{type(err).__name__}: {err}"
        print(f"[voice] synthesis failed: {err}", flush=True)
        return b""


# --- audio coming in --------------------------------------------------------


def _load_stt():
    global _stt
    with _stt_lock:
        if _stt is None:
            from faster_whisper import WhisperModel
            # int8 on CPU: about four times faster than float32 for no accuracy that
            # matters at this model size.
            _stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8",
                                download_root=str(VOICES / "whisper"))
            print(f"[voice] stt ready: {STT_MODEL}", flush=True)
    return _stt


def _transcribe(audio: bytes) -> str:
    model = _load_stt()
    # faster-whisper decodes through PyAV, which handles ogg/opus, m4a and mp3 — the
    # formats WhatsApp and Discord actually send — so no system ffmpeg is needed.
    segments, _ = model.transcribe(io.BytesIO(audio), beam_size=1)
    return " ".join(s.text for s in segments).strip()


async def listen(audio: bytes) -> str:
    """Audio in, text out. Empty string if voice is off or nothing could be heard."""
    ok, _ = available()
    if not ok or not audio:
        return ""
    try:
        return await asyncio.get_running_loop().run_in_executor(None, _transcribe, audio)
    except Exception as err:
        _state["stt_error"] = f"{type(err).__name__}: {err}"
        print(f"[voice] transcription failed: {err}", flush=True)
        return ""


# --- containers -------------------------------------------------------------


def to_ogg(wav: bytes) -> bytes:
    """WAV to OGG/Opus, for platforms that only accept that as a voice note.

    WhatsApp will attach a WAV as a file but will only render a playable voice note from
    Opus, and a voice note is the whole point. PyAV arrives with faster-whisper and
    carries its own ffmpeg, so this costs no extra dependency.
    """
    try:
        import av
        src = av.open(io.BytesIO(wav))
        out_buf = io.BytesIO()
        dst = av.open(out_buf, mode="w", format="ogg")
        stream = dst.add_stream("libopus", rate=48000)
        stream.layout = "mono"
        resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        for frame in src.decode(audio=0):
            for resampled in resampler.resample(frame):
                for packet in stream.encode(resampled):
                    dst.mux(packet)
        for packet in stream.encode(None):
            dst.mux(packet)
        dst.close()
        src.close()
        return out_buf.getvalue()
    except Exception as err:
        print(f"[voice] ogg encode failed ({type(err).__name__}: {err}) — sending wav",
              flush=True)
        return b""


def duration(wav: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav)) as r:
            return r.getnframes() / float(r.getframerate())
    except Exception:
        return 0.0


def status() -> dict:
    ok, why = available()
    return {
        "enabled": ENABLED,
        "ready": ok,
        "detail": why,
        "tts_model": TTS_VOICE,
        "stt_model": STT_MODEL,
        "tts_loaded": _tts is not None,
        "stt_loaded": _stt is not None,
        "max_chars": MAX_SPEAK_CHARS,
        "errors": {k: v for k, v in _state.items() if v},
    }

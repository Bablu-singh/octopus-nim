"""Listening and speaking, locally and for free.

The octopus could always be typed at. This gives it ears and a voice: send a voice note
from WhatsApp or Discord and it becomes a task; get the answer back as audio you can play
while doing something else. That is the difference between controlling this from a phone
and merely being able to.

Both halves run on this machine, which matters as much here as it does for the models:

  Kokoro for speech       82M parameters, Apache-2.0, ~350MB of ONNX. 54 voices across
                          eight languages — English and Hindi among them — and it sounds
                          like a person rather than a speech synthesiser. About 4x
                          realtime on this CPU. Piper is kept as a lighter alternative:
                          a fifth the disk and twice the speed, but audibly synthetic
                          and English-only in its default voice.
  faster-whisper for ears small, int8, multilingual, roughly realtime. Both of those
                          choices were measured rather than assumed — see STT_MODEL.

No API, no key, no quota, nothing sent anywhere — the same constraint the providers
follow. There is no GPU on this machine and none of this needs one.

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
# 'kokoro' or 'piper'. See the engine section below for the trade.
ENGINE = os.getenv("VOICE_ENGINE", "kokoro").strip().lower()
# Kokoro voice names. 54 are available; af_/am_ are US English, bf_/bm_ British,
# hf_/hm_ Hindi. `af_heart` and `hf_alpha` are the warmest of each.
KOKORO_EN = os.getenv("VOICE_KOKORO_EN", "af_heart")
KOKORO_HI = os.getenv("VOICE_KOKORO_HI", "hf_alpha")
SPEED = float(os.getenv("VOICE_SPEED", "1.0"))
# Breathing. Kokoro takes an explicit pause after a sentence and after a clause, and its
# defaults (0.25 / 0.10) run everything together into one flat stream. Raising them is
# most of the difference between a machine reading and a person talking.
SENTENCE_PAUSE = float(os.getenv("VOICE_SENTENCE_PAUSE", "0.5"))
CLAUSE_PAUSE = float(os.getenv("VOICE_CLAUSE_PAUSE", "0.22"))
# Only used when ENGINE=piper. Any voice from rhasspy/piper-voices.
PIPER_VOICE = os.getenv("VOICE_PIPER_MODEL", "en_US-lessac-medium")
# 'small', not 'base', and multilingual, not '.en' — both chosen by measurement rather
# than by size. On the same Hindi clip, base returned Urdu script (Hindi and Urdu are
# acoustically near-identical and base cannot tell them apart) while small returned
# correct Devanagari; on English, base heard "for agents dispatched to ran locally" where
# small heard the sentence exactly. small runs at roughly realtime here, which for a voice
# note is fine — and a transcript that is wrong is worth nothing however fast it arrives.
STT_MODEL = os.getenv("VOICE_STT_MODEL", "small")
# Force a language instead of letting Whisper guess. Rarely needed with 'small'; useful
# if you only ever dictate in one language and want the detection step skipped.
STT_LANG = os.getenv("VOICE_STT_LANG", "").strip() or None
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
        import faster_whisper  # noqa: F401
        if ENGINE == "kokoro":
            import kokoro_onnx  # noqa: F401
        else:
            import piper        # noqa: F401
    except ImportError as err:
        return False, (f"missing dependency: {err.name} — "
                       f"pip install kokoro-onnx piper-tts faster-whisper")
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

    # Give every line something to breathe on. Headings, list items and table cells
    # arrive with no terminal punctuation once the markup is stripped, so the
    # synthesiser runs them together in one unbroken breath and its sentence pause has
    # nothing to attach to. A full stop per line is what turns that back into speech.
    kept = []
    for line in text.split(chr(10)):
        line = line.strip()
        if not line:
            continue
        if line[-1] not in TERMINAL:
            line += "।" if detect_lang(line) == "hi" else "."
        kept.append(line)
    text = " ".join(kept)
    if len(text) > MAX_SPEAK_CHARS:
        # Cut on a sentence boundary if there is one nearby, so it does not stop mid-word.
        cut = text[:MAX_SPEAK_CHARS]
        dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = (cut[:dot + 1] if dot > MAX_SPEAK_CHARS * 0.6 else cut) + " … "
        text += "That is the short version; the rest is in the text."
    return text


# --- speech engines ---------------------------------------------------------
# Two, because they trade different things and the right answer depends on the language.
#
#   kokoro  82M, Apache-2.0, ~350MB of ONNX. 54 voices across eight languages including
#           Hindi, and it sounds like a person rather than a speech synthesiser. About
#           4x realtime on this CPU.
#   piper   63MB, ~8x realtime, and audibly synthetic. Kept because it is half the
#           latency and a fifth the disk, which is the right trade on a small machine
#           that only ever speaks English.
#
# Kokoro is the default: the extra second per utterance buys a voice people will actually
# listen to, and Hindi, which piper's English voice cannot do at all.

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def detect_lang(text: str) -> str:
    """'hi' or 'en'. Script is the signal, since that is what decides the voice.

    Deliberately crude. Romanised Hindi reads as English and will be spoken by an English
    voice — wrong but harmless — and the alternative is a language classifier for a
    two-way decision the alphabet already answers.
    """
    return "hi" if len(DEVANAGARI.findall(text or "")) >= 3 else "en"


def _pcm_wav(samples, rate: int) -> bytes:
    """float32 in [-1, 1] to 16-bit mono WAV, which is what every caller expects."""
    import numpy as np
    clipped = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((clipped * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _fetch_kokoro() -> None:
    """Pull the two model files once. ~350MB, and there is no smaller build."""
    import urllib.request
    base = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
            "model-files-v1.0")
    VOICES.mkdir(parents=True, exist_ok=True)
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        target = VOICES / name
        if not target.exists():
            print(f"[voice] downloading {name} (this happens once) …", flush=True)
            urllib.request.urlretrieve(f"{base}/{name}", target)


def _load_kokoro():
    global _tts
    with _tts_lock:
        if _tts is None:
            from kokoro_onnx import Kokoro
            _fetch_kokoro()
            _tts = Kokoro(str(VOICES / "kokoro-v1.0.onnx"),
                          str(VOICES / "voices-v1.0.bin"))
            print(f"[voice] tts ready: kokoro (en={KOKORO_EN}, hi={KOKORO_HI})", flush=True)
    return _tts


def _load_piper():
    global _tts
    with _tts_lock:
        if _tts is None:
            from piper import PiperVoice
            from piper.download_voices import download_voice
            VOICES.mkdir(parents=True, exist_ok=True)
            model = VOICES / f"{PIPER_VOICE}.onnx"
            if not model.exists():
                print(f"[voice] downloading {PIPER_VOICE} …", flush=True)
                download_voice(PIPER_VOICE, VOICES)
            _tts = PiperVoice.load(model)
            print(f"[voice] tts ready: piper ({PIPER_VOICE})", flush=True)
    return _tts


# Devanagari ends a sentence with a danda, not a full stop, so a splitter that knows
# only ASCII punctuation never breaks Hindi at all and the whole answer arrives in one
# breath. The danda and double danda are sentence ends; the ASCII set stays for
# English and for the romanised Hindi people actually type.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?:।॥])\s+")
# Terminal marks, used when adding punctuation to a line that has none.
TERMINAL = ".!?:;,।॥"
# A danda closes a Hindi sentence the way a full stop closes an English one.
FULL_STOPS = ".!?।॥"


def _synth(text: str) -> bytes:
    """Speak the text, breathing between sentences.

    Kokoro takes `sentence_pause` and `clause_pause` arguments and, in this build,
    ignores both: synthesising the same text with 0.0 and 1.0 produces byte-identical
    length. They only apply in `continuous` mode, which needs a model export that reports
    per-phoneme durations — not the published one. Piper has no pause control at all.

    So the pause is inserted here instead. Each sentence is synthesised on its own and
    joined with real silence, which is both more reliable than the parameter and more
    controllable: the gap is exactly as long as it is set to be. It costs one model call
    per sentence, and those are fractions of a second.
    """
    lang = detect_lang(text)
    pieces = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if not pieces:
        return b""

    if ENGINE == "kokoro":
        import numpy as np
        k = _load_kokoro()
        voice_name = KOKORO_HI if lang == "hi" else KOKORO_EN
        chunks, rate = [], 24000
        gap = None
        for i, piece in enumerate(chunks_iter := pieces):
            samples, rate = k.create(piece, voice=voice_name,
                                     lang="hi" if lang == "hi" else "en-us",
                                     speed=SPEED)
            chunks.append(np.asarray(samples, dtype="float32"))
            if i < len(chunks_iter) - 1:
                # A longer breath after a full stop than after a colon or a clause end.
                pause = SENTENCE_PAUSE if piece.rstrip()[-1:] in FULL_STOPS else CLAUSE_PAUSE
                chunks.append(np.zeros(int(rate * pause), dtype="float32"))
        return _pcm_wav(np.concatenate(chunks), rate)

    # piper: same idea, stitched through the wave module since it writes WAV directly.
    import numpy as np
    voice_obj = _load_piper()
    chunks, rate = [], 22050
    for i, piece in enumerate(pieces):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice_obj.synthesize_wav(piece, w)
        with wave.open(io.BytesIO(buf.getvalue())) as r:
            rate = r.getframerate()
            raw = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2")
        chunks.append(raw.astype("float32") / 32767.0)
        if i < len(pieces) - 1:
            pause = SENTENCE_PAUSE if piece.rstrip()[-1:] in FULL_STOPS else CLAUSE_PAUSE
            chunks.append(np.zeros(int(rate * pause), dtype="float32"))
    return _pcm_wav(np.concatenate(chunks), rate)


async def speak(text: str) -> bytes:
    """Text in, WAV out. Empty bytes if voice is off or synthesis fails.

    Runs in a worker thread: synthesis is CPU-bound and would otherwise stall the loop
    every agent is streaming through.
    """
    ok, _ = available()
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
    segments, info = model.transcribe(io.BytesIO(audio), beam_size=1, language=STT_LANG)
    text = " ".join(s.text for s in segments).strip()

    # Hindi spoken aloud is very often detected as Urdu — the two are close enough
    # acoustically that only the script really separates them, and a Devanagari speaker
    # getting Nastaliq back is useless. Re-run pinned to Hindi when that happens.
    if not STT_LANG and getattr(info, "language", "") == "ur":
        segments, _ = model.transcribe(io.BytesIO(audio), beam_size=1, language="hi")
        text = " ".join(s.text for s in segments).strip()
    return text


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
        "engine": ENGINE,
        "tts_model": (f"kokoro:{KOKORO_EN}/{KOKORO_HI}" if ENGINE == "kokoro"
                      else f"piper:{PIPER_VOICE}"),
        "languages": ["en", "hi"] if ENGINE == "kokoro" else ["en"],
        "stt_model": STT_MODEL,
        "tts_loaded": _tts is not None,
        "stt_loaded": _stt is not None,
        "stt_lang": STT_LANG or "auto",
        "pauses": {"sentence": SENTENCE_PAUSE, "clause": CLAUSE_PAUSE},
        "max_chars": MAX_SPEAK_CHARS,
        "errors": {k: v for k, v in _state.items() if v},
    }

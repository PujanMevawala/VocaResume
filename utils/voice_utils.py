"""
Voice interaction utilities for VocaResume.

Provides:
- record_audio(): Capture microphone audio via streamlit widget.
- speech_to_text(audio_bytes): Convert raw WAV/PCM bytes to text using Vosk (offline) if available, else fallback.
- text_to_speech(text): Synthesize speech via pyttsx3 (offline) or fallback.

Design Goals:
- Offline-first (Vosk + pyttsx3) to avoid extra API keys.
- Graceful degradation: Functions never raise hard exceptions to the UI layer; instead return None or log warnings.
- Cross-platform: Avoid system-specific dependencies beyond what libraries handle; pyttsx3 uses native drivers.

Notes:
- Vosk model download: Lazily attempts to load a small model if present under ./models/vosk or environment variable VOSK_MODEL_PATH.
- For recording, relies on "streamlit-audiorecorder" for front-end capture (browser side) and returns audio bytes.

Future Improvements:
- Add Whisper fallback (OpenAI) if user supplies API key & opts in.
- Cache multiple TTS voices; add rate/pitch controls.
"""

from __future__ import annotations
import io
import os
import json
import contextlib
import platform
import shutil
import subprocess
import threading
import tempfile
from dataclasses import dataclass
from textwrap import shorten
import re
import hashlib

import streamlit as st
import zipfile
import requests

# Thread lock for Vosk model downloads
_vosk_dl_lock = threading.Lock()

# ---------------------------
# Ensure Vosk model exists
# ---------------------------
def ensure_vosk_model():
    """Download a small Vosk model if not already present."""
    if os.getenv("VOSK_SKIP_DL", "").lower() in {"1", "true", "yes"}:
        return
    target = os.environ.get("VOSK_MODEL_PATH", "./models/vosk")
    marker = os.path.join(target, ".installed")
    if os.path.exists(marker):
        return
    with _vosk_dl_lock:
        if os.path.exists(marker):
            return
        os.makedirs(target, exist_ok=True)
        url = os.environ.get(
            "VOSK_MODEL_URL",
            "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        )
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                z.extractall(target)
            with open(marker, "w") as f:
                f.write("ok")
            print("[voice] Vosk model installed at", target)
        except Exception as e:
            print("[voice] Vosk model download failed:", e)
            return

try:
    ensure_vosk_model()
except Exception as e:
    print("Vosk model download skipped:", e)

# ---------------------------
# Import optional dependencies
# ---------------------------
VOICE_READY = True
MISSING = []

try:
    import vosk
except ImportError:
    VOICE_READY = False
    MISSING.append("vosk")

try:
    import pyttsx3
except ImportError:
    VOICE_READY = False
    MISSING.append("pyttsx3")

try:
    import pydub
except ImportError:
    VOICE_READY = False
    MISSING.append("pydub")

try:
    # Primary expected import name
    from streamlit_audiorecorder import audiorecorder  # type: ignore
except ImportError:
    # Some environments expose component simply as 'audiorecorder'
    try:
        from audiorecorder import audiorecorder  # type: ignore
    except ImportError:
        VOICE_READY = False
        MISSING.append("streamlit-audiorecorder")

if not VOICE_READY:
    print(f"❌ Voice dependencies missing: {', '.join(MISSING)}")
else:
    print("✅ All voice dependencies loaded successfully")

# ---------------------------
# Availability flags
# ---------------------------
_VOSK_AVAILABLE = "vosk" not in MISSING
_PYTTSX3_AVAILABLE = "pyttsx3" not in MISSING
_AUDIORECORDER_AVAILABLE = "streamlit-audiorecorder" not in MISSING

# Allow disabling offline TTS explicitly (Render free tier etc.)
_DISABLE_OFFLINE_TTS = os.getenv("DISABLE_OFFLINE_TTS", "").lower() in {"1", "true", "yes"}

# ---------------------------
# Dataclass for STT result
# ---------------------------
@dataclass
class STTResult:
    text: str
    confidence: float | None = None

# ---------------------------
# Lazy-loaded Vosk model cache
# ---------------------------
_VOSK_MODEL_CACHE = None

def _load_vosk_model():
    """Attempt to load (and cache) a Vosk model. Returns model or None."""
    global _VOSK_MODEL_CACHE
    if not _VOSK_AVAILABLE:
        return None
    if _VOSK_MODEL_CACHE is not None:
        return _VOSK_MODEL_CACHE
    candidate_paths = [
        os.getenv("VOSK_MODEL_PATH"),
        os.path.join("models", "vosk"),
        os.path.join("..", "models", "vosk"),
    ]
    for p in candidate_paths:
        if not p:
            continue
        if os.path.isdir(p):
            # If directory itself is not a model but contains a single model dir, descend
            try_paths = [p]
            try:
                contents = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)) and d.startswith("vosk-model")]
                if len(contents) == 1 and not any(fname.endswith('.conf') for fname in os.listdir(p)):
                    try_paths.insert(0, os.path.join(p, contents[0]))
            except Exception:
                pass
            for tp in try_paths:
                try:
                    _VOSK_MODEL_CACHE = vosk.Model(tp)  # type: ignore[attr-defined]
                    return _VOSK_MODEL_CACHE
                except Exception:
                    continue
    return None

# ---------------------------
# Audio recording
# ---------------------------
def record_audio(label: str = "🎤 Speak", instructions: str = "Click to start / stop recording") -> bytes | None:
    """Render an audio recorder widget and return WAV bytes once user stops."""
    if not _AUDIORECORDER_AVAILABLE:
        st.info("Audio recording component not installed (streamlit-audiorecorder).")
        return None
    st.caption(instructions)
    audio = audiorecorder(start_prompt=label, stop_prompt="⏹️ Stop")
    if len(audio) == 0:
        return None
    buf = io.BytesIO()
    try:
        audio.export(buf, format="wav")  # type: ignore
        return buf.getvalue()
    except Exception:
        try:
            raw = audio.raw_data  # type: ignore
            import wave
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(audio.channels)  # type: ignore
                wf.setsampwidth(audio.sample_width)  # type: ignore
                wf.setframerate(audio.frame_rate)  # type: ignore
                wf.writeframes(raw)
            return buf.getvalue()
        except Exception:
            st.error("Audio export failed (missing ffmpeg). Install ffmpeg or disable voice recording.")
            return None

# ---------------------------
# Speech-to-text
# ---------------------------
def speech_to_text(audio_bytes: bytes) -> STTResult | None:
    """Transcribe audio bytes using Vosk if available. Returns STTResult or None."""
    if not audio_bytes:
        return None
    if not _VOSK_AVAILABLE:
        st.warning("Vosk not installed. Install 'vosk' for offline speech recognition.")
        return None

    model = _load_vosk_model()
    if model is None:
        st.warning("Vosk model not found. Place model under models/vosk")
        return None

    try:
        import wave
        wf = wave.open(io.BytesIO(audio_bytes), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000, 16000, 32000, 44100, 48000):
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            wf.close()
            wf = wave.open(io.BytesIO(buf.getvalue()), "rb")

        rec = vosk.KaldiRecognizer(model, wf.getframerate())  # type: ignore[attr-defined]
        transcript_parts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):  # type: ignore[attr-defined]
                res = json.loads(rec.Result())  # type: ignore[attr-defined]
                if res.get("text"):
                    transcript_parts.append(res.get("text"))
        final_res = json.loads(rec.FinalResult())  # type: ignore[attr-defined]
        if final_res.get("text"):
            transcript_parts.append(final_res.get("text"))
        wf.close()
        full_text = " ".join(transcript_parts).strip()
        if not full_text:
            return None
        return STTResult(text=full_text)
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")
        return None

# ---------------------------
# Text-to-speech
# ---------------------------
def _init_tts_engine():
    if _DISABLE_OFFLINE_TTS or not _PYTTSX3_AVAILABLE:
        return None
    try:
        driver_name = 'nsss' if platform.system().lower() == 'darwin' else None
        engine = pyttsx3.init(driverName=driver_name) if driver_name else pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', min(rate, 185))
        return engine
    except Exception:
        return None

_TTS_ENGINE = _init_tts_engine()

def text_to_speech(text: str) -> bytes | None:
    """Generate spoken audio for given text. Returns WAV bytes or None."""
    if not text:
        return None
    if _TTS_ENGINE is None:
        _browser_fallback(text)
        if _PYTTSX3_AVAILABLE:
            st.warning("Local TTS engine unavailable; using browser speech instead.")
        return None

    fname = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            fname = tmp.name
        _TTS_ENGINE.save_to_file(text, fname)  # type: ignore
        _TTS_ENGINE.runAndWait()

        # Normalize audio if needed
        try:
            with open(fname, 'rb') as f:
                header = f.read(12)
            if not (header.startswith(b'RIFF') and b'WAVE' in header):
                from pydub import AudioSegment
                ffmpeg_path = shutil.which('ffmpeg')
                if ffmpeg_path:
                    AudioSegment.converter = ffmpeg_path
                seg = AudioSegment.from_file(fname)
                seg = seg.set_channels(1).set_frame_rate(22050)
                seg.export(fname, format='wav')
        except Exception:
            pass

        with open(fname, 'rb') as f:
            data = f.read()
        if data and len(data) > 512:
            return data
    except Exception as e:
        st.error(f"TTS synthesis failed: {e}")
        _browser_fallback(text, suppress_msg=True)
    finally:
        if fname and os.path.exists(fname):
            with contextlib.suppress(Exception):
                os.remove(fname)
    return None

def _browser_fallback(text: str, suppress_msg: bool = False) -> None:
    """Use browser SpeechSynthesis API for TTS playback."""
    try:
        safe = shorten(text, width=800, placeholder="…")
        st.components.v1.html(
            f"""
            <script>
            (function() {{
              if (!('speechSynthesis' in window)) return;
              const u = new SpeechSynthesisUtterance({json.dumps(safe)});
              u.rate = 1; u.pitch = 1; u.volume = 1;
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(u);
            }})();
            </script>
            """,
            height=0,
        )
        if not suppress_msg and not st.session_state.get("__browser_tts_notice", False):
            st.caption("(Browser speech synthesis used)")
            st.session_state["__browser_tts_notice"] = True
    except Exception:
        if not suppress_msg:
            st.info("Speech playback unavailable.")

# ---------------------------
# Voice utilities
# ---------------------------
def voice_enabled() -> bool:
    """Check if voice stack is functional."""
    if not _AUDIORECORDER_AVAILABLE:
        return False
    if _VOSK_AVAILABLE and _load_vosk_model() is not None:
        return True
    if _TTS_ENGINE is not None:
        return True
    return True  # Browser fallback allowed

def voice_stack_report() -> dict:
    """Return diagnostics about the voice stack."""
    return {
        "audiorecorder": _AUDIORECORDER_AVAILABLE,
        "vosk_lib": _VOSK_AVAILABLE,
        "vosk_model_loaded": bool(_load_vosk_model()) if _VOSK_AVAILABLE else False,
        "pyttsx3_lib": _PYTTSX3_AVAILABLE,
        "offline_tts_disabled": _DISABLE_OFFLINE_TTS,
        "tts_engine_ready": _TTS_ENGINE is not None,
        "browser_tts": True,
    }

# ---------------------------
# Query Optimization & Intent Routing
# ---------------------------
_FILLER_PATTERN = re.compile(r"\b(um+|uh+|like|you know|actually|basically|literally|hmm+)\b", re.IGNORECASE)

def optimize_query(text: str) -> str:
    """Lightweight normalization & filler removal to create cleaner query for LLM routing.
    - remove filler words
    - collapse whitespace
    - ensure first letter capitalized
    """
    if not text:
        return ""
    t = _FILLER_PATTERN.sub("", text)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return text.strip()
    return t[0].upper() + t[1:]

_INTENT_KEYWORDS = {
    "analysis": ["analyze", "analysis", "review", "breakdown"],
    "interview": ["interview", "question", "questions", "prep"],
    "suggestions": ["improve", "improvement", "optimize", "suggest", "suggestions"],
    "job_fit": ["fit", "score", "match", "matching"],
}

def route_query(query: str) -> tuple[int | None, str]:
    """Return (task_index or None, intent_label).
    If no strong match -> (None, 'freeform').
    """
    if not query:
        return 0, "analysis"
    q = query.lower()
    scores = {}
    for intent, kws in _INTENT_KEYWORDS.items():
        score = sum(1 for k in kws if re.search(rf"\b{re.escape(k)}\b", q))
        if score:
            scores[intent] = score
    if not scores:
        return None, "freeform"
    # pick highest score; tie-breaker stable by insertion order
    intent = max(scores.items(), key=lambda x: x[1])[0]
    mapping = {"analysis":0, "interview":1, "suggestions":2, "job_fit":3}
    return mapping.get(intent, 0), intent

# ---------------------------
# Audio Generation (MP3)
# ---------------------------
_AUDIO_CACHE: dict[str, bytes] = {}

def _tts_to_wav(text: str) -> bytes | None:
    """Internal: get WAV bytes using existing text_to_speech path (returns wav or None)."""
    return text_to_speech(text)

def _wav_to_mp3(wav_bytes: bytes) -> bytes | None:
    if not wav_bytes:
        return None
    try:
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        # light normalization
        seg = seg.set_frame_rate(22050).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="mp3", bitrate="64k")
        return buf.getvalue()
    except Exception:
        return None

def generate_audio(text: str, friendly: bool = True, segment: bool = True, max_segment_chars: int = 900) -> list[dict]:
    """Generate MP3 audio segments for response.
    Returns list of dicts: [{ 'mp3': bytes, 'text': segment_text, 'index': i }]
    Caches by hash of segment text.
    If conversion to mp3 fails, returns empty list (UI can fallback to browser TTS).
    """
    if not text:
        return []
    # Friendly tone: inject minor pauses (basic heuristic) before splitting.
    friendly_text = text
    if friendly:
        friendly_text = re.sub(r"(\n+)", " \1", friendly_text)
        # ensure sentences have slight pause markers (non audible but grouping for segmentation)
    # segmentation
    parts: list[str] = []
    if segment and len(friendly_text) > max_segment_chars:
        current = []
        chars = 0
        for sent in re.split(r"(?<=[.!?])\s+", friendly_text):
            if chars + len(sent) > max_segment_chars and current:
                parts.append(" ".join(current).strip())
                current = [sent]
                chars = len(sent)
            else:
                current.append(sent)
                chars += len(sent)
        if current:
            parts.append(" ".join(current).strip())
    else:
        parts = [friendly_text.strip()]
    outputs = []
    for i, p in enumerate(parts):
        h = hashlib.md5(p.encode("utf-8")).hexdigest()
        if h in _AUDIO_CACHE:
            outputs.append({"mp3": _AUDIO_CACHE[h], "text": p, "index": i})
            continue
        wav_bytes = _tts_to_wav(p)
        mp3_bytes = _wav_to_mp3(wav_bytes) if wav_bytes else None
        if mp3_bytes:
            _AUDIO_CACHE[h] = mp3_bytes
            outputs.append({"mp3": mp3_bytes, "text": p, "index": i})
    return outputs


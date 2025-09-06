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

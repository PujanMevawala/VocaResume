"""Voice interaction utilities for VocaResume.

Provides:
- record_audio(): Capture microphone audio via streamlit widget.
- speech_to_text(audio_bytes): Convert raw WAV/PCM bytes to text using Vosk (offline) if available, else fallback.
- text_to_speech(text): Synthesize speech via pyttsx3 (offline) or fallback to no-op.

Design Goals:
- Offline-first (Vosk + pyttsx3) to avoid extra API keys.
- Graceful degradation: Functions never raise hard exceptions to the UI layer; instead return None or log warnings.
- Cross-platform: Avoid system-specific dependencies beyond what libraries handle; pyttsx3 uses native drivers.

Notes:
- Vosk model download: We lazily attempt to load a small model if present under ./models/vosk or environment variable VOSK_MODEL_PATH.
  If not found, we instruct the user how to download but still fail softly.
- For recording, we rely on the "streamlit-audiorecorder" component to simplify front-end capture (browser side) and return audio bytes.

Future Improvements:
- Add Whisper fallback (OpenAI) if user supplies API key & opts in.
- Cache multiple TTS voices; add rate/pitch controls.
"""
from __future__ import annotations
import io
import os
import json
import contextlib
from dataclasses import dataclass
import platform
import shutil
import subprocess
from typing import Optional, Tuple
import streamlit as st
import os, zipfile, io, requests, threading
from textwrap import shorten
from .stt_providers import select_stt_provider, STTOutput

_vosk_dl_lock = threading.Lock()

def ensure_vosk_model():
    """Download a small Vosk model if not already present.

    Env Vars:
      VOSK_MODEL_PATH : target directory (default ./models/vosk)
      VOSK_MODEL_URL  : override download URL (zip)
      VOSK_SKIP_DL    : if set to 1/true, skip download
    """
    if os.getenv("VOSK_SKIP_DL", "").lower() in {"1","true","yes"}:
        return
    target = os.environ.get("VOSK_MODEL_PATH", "./models/vosk")
    marker = os.path.join(target, ".installed")
    if os.path.exists(marker):
        return
    with _vosk_dl_lock:
        if os.path.exists(marker):  # double-check
            return
        os.makedirs(target, exist_ok=True)
        url = os.environ.get("VOSK_MODEL_URL", "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                z.extractall(target)
            with open(marker, "w") as f:
                f.write("ok")
            print("[voice] Vosk model installed at", target)
        except Exception as e:
            # Leave a partial marker to avoid retry storms within same run
            print("[voice] Vosk model download failed:", e)
            return

try:
    ensure_vosk_model()
except Exception as e:
    print("Vosk model download skipped:", e)
# Optional imports guarded
try:  # Audio recording widget
    from audiorecorder import audiorecorder  # type: ignore
    _AUDIORECORDER_AVAILABLE = True
except Exception:
    _AUDIORECORDER_AVAILABLE = False

try:  # Vosk STT (optional)
    import vosk  # type: ignore
    _VOSK_AVAILABLE = True
except Exception:
    _VOSK_AVAILABLE = False

try:  # Offline TTS
    import pyttsx3  # type: ignore
    _PYTTSX3_AVAILABLE = True
except Exception:
    _PYTTSX3_AVAILABLE = False

# Allow disabling offline TTS explicitly (Render free tier etc.)
_DISABLE_OFFLINE_TTS = os.getenv("DISABLE_OFFLINE_TTS", "").lower() in {"1","true","yes"}


@dataclass
class STTResult:
    text: str
    confidence: Optional[float] = None


_VOSK_MODEL_CACHE = None  # Lazy-loaded model


def _load_vosk_model():
    """Attempt to load (and cache) a Vosk model. Returns model or None."""
    global _VOSK_MODEL_CACHE
    if not _VOSK_AVAILABLE:
        return None
    if _VOSK_MODEL_CACHE is not None:
        return _VOSK_MODEL_CACHE
    # Determine model path
    candidate_paths = [
        os.getenv("VOSK_MODEL_PATH"),
        os.path.join("models", "vosk"),
        os.path.join("..", "models", "vosk"),
    ]
    for p in candidate_paths:
        if not p:
            continue
        if os.path.isdir(p):
            try:
                _VOSK_MODEL_CACHE = vosk.Model(p)  # type: ignore[attr-defined]
                return _VOSK_MODEL_CACHE
            except Exception:
                continue
    return None


def record_audio(label: str = "🎤 Speak", instructions: str = "Click to start / stop recording") -> Optional[bytes]:
    """Render an audio recorder widget and return WAV bytes once user stops.

    Returns None if component unavailable or user hasn't recorded.
    """
    if not _AUDIORECORDER_AVAILABLE:
        st.info("Audio recording component not installed (streamlit-audiorecorder).")
        return None
    st.caption(instructions)
    audio = audiorecorder(start_prompt=label, stop_prompt="⏹️ Stop")
    if len(audio) == 0:
        return None
    # Component returns pydub.AudioSegment; attempt export
    buf = io.BytesIO()
    try:
        audio.export(buf, format="wav")  # type: ignore
        return buf.getvalue()
    except Exception:
        # If export fails (likely missing ffmpeg), attempt raw parameters
        try:
            raw = audio.raw_data  # type: ignore
            # Construct a minimal WAV header
            import wave, struct
            tmp = io.BytesIO()
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(audio.channels)  # type: ignore
                wf.setsampwidth(audio.sample_width)  # type: ignore
                wf.setframerate(audio.frame_rate)  # type: ignore
                wf.writeframes(raw)
            return tmp.getvalue()
        except Exception:
            st.error("Audio export failed (missing ffmpeg). Install ffmpeg or disable voice recording.")
            return None


def speech_to_text(audio_bytes: bytes) -> Optional[STTResult]:
    """Transcribe audio using first available provider (Whisper API, local Vosk, etc.)."""
    if not audio_bytes:
        return None
    stt_runner = select_stt_provider()
    out: Optional[STTOutput] = stt_runner(audio_bytes)
    if out:
        return STTResult(text=out.text, confidence=None)
    st.info("No speech-to-text provider available or transcription empty.")
    return None


def _init_tts_engine():
    if _DISABLE_OFFLINE_TTS or not _PYTTSX3_AVAILABLE:
        return None
    try:
        # Use native NSSpeechSynthesizer on macOS for reliability
        driver_name = 'nsss' if platform.system().lower() == 'darwin' else None
        engine = pyttsx3.init(driverName=driver_name) if driver_name else pyttsx3.init()
        # Slightly slower rate for clarity
        rate = engine.getProperty('rate')
        engine.setProperty('rate', min(rate, 185))
        return engine
    except Exception:
        return None

_TTS_ENGINE = _init_tts_engine()


def text_to_speech(text: str, format: str = "wav") -> Optional[bytes]:
    """Generate spoken audio for given text.

    Order:
      1. pyttsx3 (offline) → WAV
      2. gTTS (MP3) if installed
      3. Browser speech fallback (no bytes)
    """
    if not text:
        return None
    # Offline engine
    if _TTS_ENGINE is not None and format == "wav":
        import tempfile
        fname = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                fname = tmp.name
            _TTS_ENGINE.save_to_file(text, fname)  # type: ignore
            _TTS_ENGINE.runAndWait()
            with open(fname, 'rb') as f:
                data = f.read()
            return data if data and len(data) > 512 else None
        except Exception:
            pass
        finally:
            if fname and os.path.exists(fname):
                with contextlib.suppress(Exception):
                    os.remove(fname)
    # gTTS path (MP3)
    try:
        from gtts import gTTS  # type: ignore
        import tempfile
        mp3_buf = io.BytesIO()
        gTTS(text).write_to_fp(mp3_buf)
        return mp3_buf.getvalue()
    except Exception:
        _browser_fallback(text)
        return None


def _browser_fallback(text: str, suppress_msg: bool = False) -> None:
    """Inject a small snippet that uses the browser's SpeechSynthesis if available.
    No return (client plays speech). Safe no-op if components disabled.
    """
    try:
        safe = shorten(text, width=800, placeholder="…")  # avoid gigantic injection
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


def voice_enabled() -> bool:
    """Voice considered enabled if we can at least record and provide either STT or any TTS path.
    Browser-only TTS counts when offline engine disabled."""
    if not _AUDIORECORDER_AVAILABLE:
        return False
    # STT path
    if _VOSK_AVAILABLE and _load_vosk_model() is not None:
        return True
    # Offline TTS engine
    if _TTS_ENGINE is not None:
        return True
    # Browser fallback allowed
    return True  # recorder + browser speech


def voice_stack_report() -> dict:
    """Return diagnostics about the voice stack for UI display."""
    return {
        "audiorecorder": _AUDIORECORDER_AVAILABLE,
        "vosk_lib": _VOSK_AVAILABLE,
        "vosk_model_loaded": bool(_load_vosk_model()) if _VOSK_AVAILABLE else False,
        "pyttsx3_lib": _PYTTSX3_AVAILABLE,
        "offline_tts_disabled": _DISABLE_OFFLINE_TTS,
        "tts_engine_ready": _TTS_ENGINE is not None,
        "browser_tts": True,
    }

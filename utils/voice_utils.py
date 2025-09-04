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

# Optional imports guarded
try:  # Audio recording widget
    from audiorecorder import audiorecorder  # type: ignore
    _AUDIORECORDER_AVAILABLE = True
except Exception:
    _AUDIORECORDER_AVAILABLE = False

try:  # Vosk STT
    import vosk  # type: ignore
    _VOSK_AVAILABLE = True
except Exception:
    _VOSK_AVAILABLE = False

try:  # Offline TTS
    import pyttsx3  # type: ignore
    _PYTTSX3_AVAILABLE = True
except Exception:
    _PYTTSX3_AVAILABLE = False


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
    # Component returns pydub.AudioSegment; export to wav bytes
    buf = io.BytesIO()
    audio.export(buf, format="wav")  # type: ignore
    return buf.getvalue()


def speech_to_text(audio_bytes: bytes) -> Optional[STTResult]:
    """Transcribe audio bytes using Vosk if available. Returns STTResult or None.

    If unavailable, returns None and displays a warning.
    """
    if not audio_bytes:
        return None
    if not _VOSK_AVAILABLE:
        st.warning("Vosk not installed. Install 'vosk' for offline speech recognition.")
        return None

    model = _load_vosk_model()
    if model is None:
        st.warning("Vosk model not found. Download a small model (e.g., 'vosk-model-small-en-us-0.15') and place it under models/vosk")
        return None

    try:
        import wave
        wf = wave.open(io.BytesIO(audio_bytes), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000, 16000, 32000, 44100, 48000):
            # Convert audio using pydub if format unexpected
            try:
                from pydub import AudioSegment  # type: ignore
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                buf = io.BytesIO()
                seg.export(buf, format="wav")
                wf = wave.open(io.BytesIO(buf.getvalue()), "rb")
            except Exception:
                st.error("Failed to normalize audio for STT.")
                return None

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
        full_text = " ".join(transcript_parts).strip()
        if not full_text:
            return None
        return STTResult(text=full_text)
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")
        return None


def _init_tts_engine():
    if not _PYTTSX3_AVAILABLE:
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


def text_to_speech(text: str) -> Optional[bytes]:
    """Generate spoken audio for given text. Returns WAV bytes or None.

    If TTS engine not available, returns None and surfaces info message.
    """
    if not text:
        return None
    if _TTS_ENGINE is None:
        if _PYTTSX3_AVAILABLE:
            st.error("Failed to init pyttsx3 engine.")
        else:
            st.info("pyttsx3 not installed. Install it for offline text-to-speech.")
        return None

    # pyttsx3 saves to a file; ensure WAV output and normalize if driver writes AIFF
    import tempfile
    fname = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            fname = tmp.name
        _TTS_ENGINE.save_to_file(text, fname)  # type: ignore
        _TTS_ENGINE.runAndWait()
        # Some platforms may write non-wav; normalize via pydub if needed
        try:
            with open(fname, 'rb') as f:
                data12 = f.read(12)
            if not (data12.startswith(b'RIFF') and b'WAVE' in data12):
                from pydub import AudioSegment  # type: ignore
                ffmpeg_path = shutil.which('ffmpeg')
                if ffmpeg_path:
                    AudioSegment.converter = ffmpeg_path  # type: ignore[attr-defined]
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
    finally:
        if fname and os.path.exists(fname):
            with contextlib.suppress(Exception):
                os.remove(fname)

    # Fallback: macOS 'say' to AIFF then convert to WAV
    try:
        if platform.system().lower() == 'darwin' and shutil.which('say'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.aiff') as aiff_tmp:
                aiff_path = aiff_tmp.name
            subprocess.run(['say', '-o', aiff_path, text], check=True)
            wav_path = aiff_path.replace('.aiff', '.wav')
            converted = False
            try:
                from pydub import AudioSegment  # type: ignore
                ffmpeg_path = shutil.which('ffmpeg')
                if ffmpeg_path:
                    AudioSegment.converter = ffmpeg_path  # type: ignore[attr-defined]
                seg = AudioSegment.from_file(aiff_path)
                seg = seg.set_channels(1).set_frame_rate(22050)
                seg.export(wav_path, format='wav')
                converted = True
            except Exception:
                if shutil.which('afconvert'):
                    try:
                        subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@22050', aiff_path, wav_path], check=True)
                        converted = True
                    except Exception:
                        converted = False
            if converted and os.path.exists(wav_path):
                with open(wav_path, 'rb') as f:
                    data = f.read()
                with contextlib.suppress(Exception):
                    os.remove(aiff_path)
                    os.remove(wav_path)
                if data and len(data) > 512:
                    return data
            with open(aiff_path, 'rb') as f:
                data = f.read()
            with contextlib.suppress(Exception):
                os.remove(aiff_path)
            return data if data and len(data) > 512 else None
    except Exception as e:
        st.error(f"macOS TTS fallback failed: {e}")
        return None


def voice_enabled() -> bool:
    """Helper to check if minimal voice stack is available (record + stt + tts)."""
    return _AUDIORECORDER_AVAILABLE and (_VOSK_AVAILABLE or _PYTTSX3_AVAILABLE)

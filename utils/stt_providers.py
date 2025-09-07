"""Speech-to-Text provider abstraction layer.

Goal: Offer deploy-friendly STT without forcing a heavy local Vosk model.

Priority order (first available used):
1. OpenAI Whisper API (if OPENAI / PPLX compatible key present and WHISPER_API_ENABLED=1)
2. Faster-Whisper local (if installed) small model (optional future enhancement)
3. Vosk (existing) if model folder detected
4. Fallback: no STT (returns None)

Design:
- Each provider implements a function taking WAV/bytes -> text or None.
- select_stt_provider() returns a callable for use by voice_utils.

Environment Flags:
WHISPER_API_ENABLED=1  Enable remote Whisper API call (cost applies per provider pricing).
WHISPER_MODEL=whisper-1  Override model name.

Future: Add Google Speech (speech_recognition) integration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import os, io, wave

TARGET_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit

@dataclass
class STTOutput:
    text: str
    provider: str

# --- Provider Implementations --- #

def _try_whisper_api(audio_bytes: bytes) -> Optional[STTOutput]:
    if os.getenv("WHISPER_API_ENABLED", "").lower() not in {"1","true","yes"}:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PPLX_API_KEY") or os.getenv("PPLX_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        model = os.getenv("WHISPER_MODEL", "whisper-1")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                resp = client.audio.transcriptions.create(model=model, file=f)  # type: ignore
        text = (resp.text or "").strip()
        if text:
            return STTOutput(text=text, provider="whisper_api")
    except Exception:
        return None
    return None


def _normalize_for_vosk(raw: bytes) -> Optional[bytes]:
    """Ensure WAV mono 16k 16-bit for Vosk. Returns normalized WAV bytes or None."""
    try:
        buf = io.BytesIO(raw)
        wf = wave.open(buf, 'rb')
        params_ok = (wf.getnchannels() == TARGET_CHANNELS and wf.getframerate() == TARGET_RATE and wf.getsampwidth() == TARGET_SAMPLE_WIDTH)
        wf.close()
        if params_ok:
            return raw
        # Convert via pydub if available
        try:
            from pydub import AudioSegment  # type: ignore
            seg = AudioSegment.from_file(io.BytesIO(raw))
            seg = seg.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_RATE).set_sample_width(TARGET_SAMPLE_WIDTH)
            out = io.BytesIO()
            seg.export(out, format='wav')
            return out.getvalue()
        except Exception:
            return raw  # fallback; let Vosk try
    except Exception:
        return raw


def _try_vosk(audio_bytes: bytes) -> Optional[STTOutput]:
    model_path = os.getenv("VOSK_MODEL_PATH", "./models/vosk")
    if not os.path.isdir(model_path):
        return None
    try:
        import json, vosk  # type: ignore
        norm = _normalize_for_vosk(audio_bytes) or audio_bytes
        wf = wave.open(io.BytesIO(norm), "rb")
        model = vosk.Model(model_path)  # type: ignore
        rec = vosk.KaldiRecognizer(model, wf.getframerate())  # type: ignore
        parts = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):  # type: ignore
                r = json.loads(rec.Result())
                if r.get("text"):
                    parts.append(r["text"])
        final = json.loads(rec.FinalResult())
        if final.get("text"):
            parts.append(final["text"])
        text = " ".join(parts).strip()
        if text:
            return STTOutput(text=text, provider="vosk")
    except Exception:
        return None
    return None

# Placeholder for future local faster-whisper

def select_stt_provider() -> Callable[[bytes], Optional[STTOutput]]:
    def _runner(audio: bytes) -> Optional[STTOutput]:
        for fn in (_try_whisper_api, _try_vosk):
            out = fn(audio)
            if out:
                return out
        return None
    return _runner

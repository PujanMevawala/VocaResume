"""Simplified TTS utilities for VocaResume.

This module now ONLY provides Text-To-Speech functionality using Edge TTS as the
primary provider with graceful fallbacks (gTTS -> pyttsx3 -> browser speech).

Removed legacy Speech-To-Text and recording code; the application now relies on
pure text input.
"""
from __future__ import annotations
import io
import os
import json
import contextlib
import platform
import shutil
import subprocess
from typing import Optional, Tuple, Union
import asyncio
import tempfile
import logging
import uuid
from pathlib import Path
import streamlit as st
import os, zipfile, io, requests, threading
from textwrap import shorten
from .text_utils import normalize_for_tts

_AUDIORECORDER_AVAILABLE = False
_VOSK_AVAILABLE = False

try:  # Offline TTS (legacy fallback)
    import pyttsx3  # type: ignore
    _PYTTSX3_AVAILABLE = True
except Exception:
    _PYTTSX3_AVAILABLE = False

try:  # Edge TTS
    import edge_tts  # type: ignore
    _EDGETTS_AVAILABLE = True
except Exception:
    _EDGETTS_AVAILABLE = False

# Allow disabling offline TTS explicitly (Render free tier etc.)
_DISABLE_OFFLINE_TTS = os.getenv("DISABLE_OFFLINE_TTS", "").lower() in {"1","true","yes"}
VOICE_DEBUG = os.getenv("VOICE_DEBUG", "0").lower() in {"1","true","yes"}

logger = logging.getLogger("voice")
if VOICE_DEBUG and not logger.handlers:
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
else:
    logger.addHandler(logging.NullHandler())


## STT & recording removed.


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


async def generate_tts_from_text(text: str, voice: str = "en-US-JennyNeural") -> Optional[Path]:
    """Generate an MP3 file (EdgeTTS preferred) from text.

    Returns path to mp3 or None. Uses sanitize (markdown → plain) via normalize_for_tts.
    Steps:
      1. Sanitize text
      2. EdgeTTS synth to temporary .mp3.part
      3. Normalize sample rate to 22050 Hz using pydub/ffmpeg
      4. Atomic rename to final .mp3
      5. Fallback to gTTS (sync) or pyttsx3 if EdgeTTS unavailable or fails
    """
    if not text:
        return None
    is_ssml = text.strip().lower().startswith('<speak>')
    sanitized = text if is_ssml else normalize_for_tts(text)
    base_dir = Path(tempfile.gettempdir()) / "vocaresume_tts"
    base_dir.mkdir(parents=True, exist_ok=True)
    final_path = base_dir / f"{uuid.uuid4().hex}.mp3"
    part_path = final_path.with_suffix('.mp3.part')

    # Primary: Edge TTS
    if _EDGETTS_AVAILABLE:
        try:
            logger.debug(f"EdgeTTS synth start voice={voice} chars={len(sanitized)}")
            communicate = edge_tts.Communicate(sanitized, voice)  # type: ignore
            with part_path.open('wb') as f:
                async for chunk in communicate.stream():  # type: ignore
                    if chunk.get('type') == 'audio':
                        f.write(chunk.get('data', b''))
            # Normalize with pydub if possible
            try:
                from pydub import AudioSegment  # type: ignore
                audio = AudioSegment.from_file(part_path, format='mp3')
                audio = audio.set_frame_rate(22050)
                audio.export(part_path, format='mp3', bitrate='64k')
            except Exception as e:  # pragma: no cover - best effort
                logger.debug(f"pydub normalization skipped: {e}")
            part_path.replace(final_path)
            logger.debug(f"EdgeTTS synth success -> {final_path}")
            return final_path
        except Exception as e:
            logger.warning(f"EdgeTTS failed: {e}")

    # Fallback: gTTS
    try:
        from gtts import gTTS  # type: ignore
        from pydub import AudioSegment  # type: ignore
        tmp_mp3 = part_path
        gTTS(sanitized).save(str(tmp_mp3))
        try:
            audio = AudioSegment.from_file(tmp_mp3, format='mp3')
            audio = audio.set_frame_rate(22050)
            audio.export(tmp_mp3, format='mp3', bitrate='64k')
        except Exception:
            pass
        tmp_mp3.replace(final_path)
        logger.info("gTTS fallback used")
        return final_path
    except Exception as e:
        logger.warning(f"gTTS fallback failed: {e}")

    # Legacy final fallback: pyttsx3 to wav then convert
    if _TTS_ENGINE is not None:
        try:
            from pydub import AudioSegment  # type: ignore
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpw:
                wav_name = tmpw.name
            _TTS_ENGINE.save_to_file(sanitized, wav_name)  # type: ignore
            _TTS_ENGINE.runAndWait()
            audio = AudioSegment.from_wav(wav_name)
            audio = audio.set_frame_rate(22050)
            audio.export(part_path, format='mp3', bitrate='64k')
            Path(wav_name).unlink(missing_ok=True)
            part_path.replace(final_path)
            logger.info("pyttsx3 fallback used")
            return final_path
        except Exception as e:  # pragma: no cover
            logger.error(f"pyttsx3 fallback failed: {e}")
    return None


def text_to_speech(text: str, format: str = "mp3") -> Optional[bytes]:
    """Synchronous wrapper returning MP3 bytes for compatibility.

    Internally invokes EdgeTTS async pipeline. If EdgeTTS unavailable, falls back
    to legacy cascade. Returns None if all providers fail; browser fallback triggered.
    """
    if not text:
        return None
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            path = loop.run_until_complete(generate_tts_from_text(text))
        finally:
            loop.close()
        if path and path.exists():
            data = path.read_bytes()
            # optionally clean up older files to avoid disk bloat
            try:
                for f in path.parent.glob('*.mp3'):
                    if f != path and f.stat().st_mtime < path.stat().st_mtime - 3600:
                        f.unlink(missing_ok=True)
            except Exception:
                pass
            return data if data and len(data) > 512 else None
    except Exception as e:
        logger.error(f"EdgeTTS sync wrapper failed: {e}")
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


def voice_enabled() -> bool:  # Simplified definition
    return _EDGETTS_AVAILABLE or _PYTTSX3_AVAILABLE


def voice_stack_report() -> dict:
    return {
        "edge_tts": _EDGETTS_AVAILABLE,
        "pyttsx3_lib": _PYTTSX3_AVAILABLE,
        "offline_tts_disabled": _DISABLE_OFFLINE_TTS,
        "tts_engine_ready": _TTS_ENGINE is not None,
    }

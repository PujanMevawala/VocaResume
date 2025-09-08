import asyncio
from pathlib import Path
import sys
import pytest

from utils import voice_utils

class FakeEdgeChunkStream:
    def __init__(self, data: bytes):
        self.data = data
        self._yielded = False
    async def __aiter__(self):
        if not self._yielded:
            self._yielded = True
            yield {"type": "audio", "data": self.data}
        return

class FakeCommunicate:
    def __init__(self, text, voice):
        self.text = text
        self.voice = voice
    async def stream(self):
        # produce very small fake mp3 header + data (not a valid file but good for flow)
        # Use bytes that pydub may ignore; we won't rely on decoding for the test.
        async for c in FakeEdgeChunkStream(b"ID3fakeMP3data"):
            yield c

@pytest.mark.asyncio
async def test_generate_tts_edge_mock(monkeypatch, tmp_path):
    # Ensure edge_tts considered available
    monkeypatch.setattr(voice_utils, '_EDGETTS_AVAILABLE', True)
    # Monkeypatch edge_tts.Communicate
    class DummyModule:
        @staticmethod
        def Communicate(text, voice):
            return FakeCommunicate(text, voice)
    # Provide dummy module in sys.modules so import within function scope would resolve if used
    monkeypatch.setitem(sys.modules, 'edge_tts', DummyModule)
    # No direct attribute needed on voice_utils since we only rely on availability flag.
    path = await voice_utils.generate_tts_from_text("# Heading\nSome *markdown* text")
    assert path is not None
    assert path.exists()
    assert path.suffix == '.mp3'


def test_sync_wrapper_smoke(monkeypatch):
    # Force generate_tts_from_text to return a temp file path
    import tempfile
    tmp_file = Path(tempfile.gettempdir()) / 'dummy.mp3'
    tmp_file.write_bytes(b"ID3smalldata" * 200)  # ensure > 512 bytes
    async def fake_gen(text, voice='x'):
        return tmp_file
    monkeypatch.setattr(voice_utils, 'generate_tts_from_text', fake_gen)
    # Ensure EdgeTTS availability path executes
    monkeypatch.setattr(voice_utils, '_EDGETTS_AVAILABLE', True)
    data = voice_utils.text_to_speech("Hello world")
    assert data is not None
    assert len(data) > 10

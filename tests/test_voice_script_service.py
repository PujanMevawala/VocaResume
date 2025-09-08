import json
import types
import pytest
from services import voice_script_service

class DummyHandler:
    def __call__(self, sys_text, pdf_mime, pdf_data, prompt, groq_client, pplx_client, model_name, max_tokens):
        # Return JSON with plan and script
        payload = {"plan": "1. Greet user 2. Summarize strengths 3. Give improvement", "script": "Hi there, you show strong Python skills. One area to improve is cloud depth. Overall solid foundation."}
        return json.dumps(payload)

def test_generate_voice_script_success(monkeypatch):
    monkeypatch.setitem(voice_script_service._PROVIDER_DISPATCH, 'google', DummyHandler())
    model_info = {"provider": "google", "model": "test-model"}
    result = voice_script_service.generate_voice_script("analysis", "Give me resume insights", "Strengths: Python\nWeaknesses: Cloud", model_info)
    assert result['status'] == 'success'
    assert 'Python' in result['script']
    assert len(result['script'].split()) <= voice_script_service.MAX_SCRIPT_WORDS
    assert '<' not in result['script'] or '>' not in result['script']  # sanitized

def test_generate_voice_script_short_query(monkeypatch):
    model_info = {"provider": "google", "model": "test-model"}
    result = voice_script_service.generate_voice_script("analysis", "Hi", "Strengths: Python", model_info)
    assert result['status'] == 'unavailable'

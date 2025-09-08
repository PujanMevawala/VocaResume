"""Voice script generation service.
Generates a planned conversational script separate from raw LLM analytical output.

Contract:
  generate_voice_script(task_id: str, user_query: str, raw_markdown: str, model_info: dict, groq_client, pplx_client, max_output_tokens:int) -> dict
Return dict keys:
  status: success|unavailable|error
  plan: optional high-level bullet planning (string)
  script: final conversational script (plain text; no markdown)
  ssml: SSML version (optional)
Logic:
  - If user_query too short (<4 words) -> unavailable
  - Prompt LLM with planning instructions separate from raw content.
  - Enforce brevity (~170 words) & user-centric style.
"""
from __future__ import annotations
from typing import Any, Dict
import re
from utils.text_utils import normalize_for_tts, clean_markdown
from utils.speech_adapter import build_ssml
from services.model_service import _PROVIDER_DISPATCH  # reuse provider mapping

MIN_QUERY_WORDS = 4
MAX_SCRIPT_WORDS = 180

VOICE_SYSTEM_INSTRUCTIONS = (
    "You are a helpful career assistant. Generate a concise spoken script summarizing results for audio playback. "
    "Rules: conversational, encouraging, no markdown, no angle brackets unless part of provided SSML markers, max 180 words. "
    "Include: brief strengths summary, one improvement, and an overall takeaway. If content lacks distinct areas, acknowledge that briefly."
)

SCRIPT_PROMPT_TEMPLATE = (
    "User Query: {query}\n\nRaw Analysis (do not read verbatim, only extract insights):\n" \
    "{analysis}\n\nOutput JSON with keys: plan (string bullet-like short plan) and script (final spoken text)."
)

import json


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def _strip_brackets(text: str) -> str:
    # Remove stray angle brackets but keep those forming typical SSML tags if already inserted (<emphasis>, <break) etc
    return re.sub(r"<(?!/?(emphasis|break|speak)\b)[^>]+>", "", text)


def generate_voice_script(task_id: str, user_query: str, raw_markdown: str, model_info: Dict[str,str], groq_client=None, pplx_client=None, max_output_tokens: int = 512) -> Dict[str,str]:
    if not raw_markdown:
        return {"status": "error", "message": "No analysis content"}
    if not user_query or len(user_query.split()) < MIN_QUERY_WORDS:
        return {"status": "unavailable", "message": "Query too short for voice script."}
    provider = model_info.get("provider")
    model_name = model_info.get("model")
    handler = _PROVIDER_DISPATCH.get(provider)
    if not handler:
        return {"status": "error", "message": f"Unknown provider {provider}"}
    analysis_clean = normalize_for_tts(raw_markdown, max_chars=6000)
    prompt = SCRIPT_PROMPT_TEMPLATE.format(query=user_query.strip(), analysis=analysis_clean[:4000])
    # Use the same handler but adapt to expected signature
    try:
        response_text = handler(VOICE_SYSTEM_INSTRUCTIONS, "text/plain", prompt, "Generate voice script", groq_client, pplx_client, model_name, max_output_tokens)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    # Attempt to parse JSON; fallback heuristic
    plan = ""
    script = ""
    try:
        data = json.loads(response_text)
        plan = clean_markdown(str(data.get("plan", "")).strip())
        script = clean_markdown(str(data.get("script", "")).strip())
    except Exception:
        # Heuristic split
        parts = response_text.split("\n\n", 1)
        plan = clean_markdown(parts[0].strip())
        script = clean_markdown(parts[1].strip() if len(parts) > 1 else parts[0].strip())
    script = _truncate_words(script, MAX_SCRIPT_WORDS)
    script = _strip_brackets(script)
    if not script:
        return {"status": "unavailable", "message": "Could not derive script."}
    ssml = build_ssml(script)
    return {"status": "success", "plan": plan, "script": script, "ssml": ssml}

__all__ = ["generate_voice_script"]

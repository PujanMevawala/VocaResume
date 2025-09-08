"""Script planning service that always uses Gemini 2.5 Pro.

Takes cleaned analysis text (preprocessed) and produces a conversational script
for TTS distinct from the raw text.
"""
from __future__ import annotations
import difflib
from typing import Dict
import google.generativeai as genai

PLANNER_MODEL = "gemini-2.5-pro"

PROMPT_TEMPLATE = (
    "You are a career assistant generating a brief spoken summary.\n"
    "Input (cleaned analysis, not to be read verbatim):\n{clean}\n\n"
    "Produce JSON with keys: plan (short bullet-like outline string) and script (final spoken text under 160 words).\n"
    "Rules: conversational, positive, no markdown, no lists/brackets, no code fences."
)


def plan_script(clean_text: str, api_key: str | None) -> Dict[str,str]:
    if not clean_text:
        return {"status":"error","message":"Empty clean text"}
    if not api_key:
        return {"status":"error","message":"Missing GOOGLE_API_KEY for Gemini"}
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(PLANNER_MODEL)
    prompt = PROMPT_TEMPLATE.format(clean=clean_text[:6000])
    try:
        resp = model.generate_content(prompt)
        raw = resp.text or ""
    except Exception as e:
        return {"status":"error","message":str(e)}
    plan = ""
    script = raw.strip()
    # crude JSON extraction if model enclosed
    if script.startswith('{'):
        import json
        try:
            data = json.loads(script)
            plan = str(data.get('plan','')).strip()
            script = str(data.get('script','')).strip()
        except Exception:
            pass
    # similarity check
    ratio = difflib.SequenceMatcher(None, clean_text.lower(), script.lower()).ratio()
    if ratio > 0.92:  # too similar -> force rephrase minimal tweak
        script = "Here is a concise spoken summary: " + script[:350]
    return {"status":"success","plan":plan, "script":script}

__all__ = ["plan_script"]

"""Conversational speech adapter.

Transforms raw LLM markdown/analysis into a concise, friendly, spoken-style summary.
"""
from __future__ import annotations
import re
from typing import Optional, List
from .text_utils import normalize_for_tts, clean_markdown

CONVERSATIONAL_MARKERS = [
    "here's what stood out",
    "great news",
    "you might want to",
    "overall",
]

SECTION_PATTERNS = [
    (re.compile(r"strengths?[:\-]", re.I), "strengths"),
    (re.compile(r"weaknesses|areas\s+for\s+improvement[:\-]", re.I), "weaknesses"),
    (re.compile(r"suggestions?[:\-]", re.I), "suggestions"),
    (re.compile(r"overall|summary[:\-]", re.I), "overall"),
    (re.compile(r"job\s*fit|fit\s*score[:\-]", re.I), "fit"),
]

MAX_SPOKEN_WORDS = 300  # ~1 minute at ~150-180 wpm upper safe bound


def _split_sections(text: str) -> List[str]:
    # naive split by headings or double newlines
    parts = re.split(r"\n{2,}|^#+.*$", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _extract_key_phrases(section: str) -> str:
    # take first sentence or list items condensed
    # remove bullets
    section = re.sub(r"^[\-*+]\s+", "", section, flags=re.MULTILINE)
    # first sentence heuristic
    m = re.search(r"(.{15,260}?\.)(\s|$)", section)
    if m:
        return m.group(1).strip()
    # fallback truncate
    return section[:180].rstrip() + ("…" if len(section) > 180 else "")


def generate_spoken_version(analysis_text: str, user_name: Optional[str] = None) -> str:
    if not analysis_text:
        return ""
    # strip markdown to simplify downstream processing
    plain = normalize_for_tts(analysis_text, max_chars=8000)
    sections = _split_sections(plain)
    buckets = {k: [] for k in ["strengths", "weaknesses", "suggestions", "overall", "fit", "other"]}
    for sec in sections:
        classified = False
        for pattern, label in SECTION_PATTERNS:
            if pattern.search(sec):
                buckets[label].append(sec)
                classified = True
                break
        if not classified:
            buckets["other"].append(sec)

    def join_bucket(name: str, prefix: str) -> Optional[str]:
        items = buckets.get(name) or []
        if not items:
            return None
        phrase = _extract_key_phrases(" ".join(items))
        return f"{prefix} {phrase}" if phrase else None

    intro = f"Hi {user_name}, here's what stood out to me" if user_name else "Here's what stood out to me"
    strengths_line = join_bucket("strengths", "Great news — your strengths include")
    weaknesses_line = join_bucket("weaknesses", "You might want to improve")
    fit_line = join_bucket("fit", "Regarding overall fit,")
    suggestions_line = join_bucket("suggestions", "Some suggestions:")
    overall_line = join_bucket("overall", "Overall,") or join_bucket("other", "Overall,")

    ordered = [intro, strengths_line, weaknesses_line, fit_line, suggestions_line, overall_line]
    spoken = ". ".join([p for p in ordered if p])
    spoken = re.sub(r"\s+", " ", spoken).strip()

    # word limit
    words = spoken.split()
    if len(words) > MAX_SPOKEN_WORDS:
        words = words[:MAX_SPOKEN_WORDS - 5] + ["…"]
        spoken = " ".join(words)

    # ensure conversational markers exist
    if not any(m in spoken.lower() for m in CONVERSATIONAL_MARKERS):
        spoken = intro + ". " + spoken
    return spoken


def build_ssml(spoken_text: str) -> str:
    if not spoken_text:
        return ""
    safe = clean_markdown(spoken_text)
    # Remove stray angle brackets that are not approved SSML tags
    safe = re.sub(r"<(?!/?(emphasis|break|speak)\b)[^>]+>", "", safe)
    # emphasis key terms
    def emph(word: str) -> str:
        return re.sub(rf"\b({word})\b", r"<emphasis>\1</emphasis>", safe, flags=re.I)
    enhanced = safe
    for w in ["strengths", "weaknesses", "overall", "suggestions", "improve"]:
        enhanced = emph(w)
    # insert breaks after sentences
    enhanced = re.sub(r"\.\s+", ". <break time=\"400ms\"/> ", enhanced)
    return f"<speak>{enhanced}</speak>"

__all__ = ["generate_spoken_version", "build_ssml"]

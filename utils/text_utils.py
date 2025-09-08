"""Text / markdown related helper utilities.

New functionality:
 - normalize_for_tts(): Convert arbitrary markdown to a readable plain text string
   optimized for Text-To-Speech (TTS) engines. Removes formatting tokens, converts
   lists / headings to simple prefixes, strips code blocks (or summarizes), collapses
   whitespace, and truncates to a safe char limit.
"""
from __future__ import annotations

import re
import html
from typing import Iterable

try:  # lightweight optional imports; only needed for normalize_for_tts
    import mistune  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
    _SANITIZE_DEPS = True
except Exception:  # pragma: no cover - if missing we degrade gracefully
    _SANITIZE_DEPS = False

_MD_STRIP_PATTERN = re.compile(r"[#*_`]+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_markdown(text: str) -> str:  # existing simple helper retained
    """Remove simple markdown formatting tokens to display as plain text.
    This is intentionally conservative; extend as needed.
    """
    if not text:
        return ""
    return _MD_STRIP_PATTERN.sub("", text)


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    # try to cut at sentence boundary within last 200 chars of window
    window = s[:max_chars]
    period_idx = window.rfind(".")
    if period_idx > max_chars - 200:
        return window[:period_idx + 1] + " …"
    return window.rstrip() + " …"


def _convert_html_to_text(soup: 'BeautifulSoup') -> str:
    lines: list[str] = []
    # Use descendants (recursiveChildGenerator deprecated in bs4 >=4)
    for el in soup.descendants:  # type: ignore[attr-defined]
        name = getattr(el, 'name', None)
        if name is None:
            # NavigableString
            txt = str(el)
            if txt.strip():
                lines.append(txt)
            continue
        if name in {"h1","h2","h3","h4","h5","h6"}:
            level = int(name[1])
            heading_txt = el.get_text(strip=True)
            if heading_txt:
                prefix = "" if level <= 2 else "- "
                lines.append(f"{prefix}{heading_txt}:")
        elif name in {"li"}:
            li_txt = el.get_text(strip=True)
            if li_txt:
                lines.append(f"• {li_txt}")
        elif name == "code":
            # Inline code: wrap in quotes for readability
            code_txt = el.get_text(strip=True)
            if code_txt:
                lines.append(f"'{code_txt}'")
        elif name == "pre":
            # Block code: summarize length only (skip verbose code for TTS)
            code_txt = el.get_text("\n", strip=True)
            if code_txt:
                # keep short snippets (<=80 chars single line)
                short = code_txt.replace('\n', ' ')
                short = _WHITESPACE_RE.sub(' ', short)
                if len(short) <= 80:
                    lines.append(f"Snippet: {short}")
                else:
                    lines.append("(code block omitted)")
        # Paragraphs handled implicitly by text nodes
    # join with spaces while preserving bullet spacing
    joined = ' '.join(part.strip() for part in lines if part.strip())
    # restore bullet list newlines lightly for clarity
    joined = re.sub(r"(?:• [^•]+)(?:\s+)(?=• )", "\n", joined)
    return joined


def normalize_for_tts(md_text: str, max_chars: int = 4800) -> str:
    """Convert markdown to a clean, TTS-friendly plain text string.

    Steps:
      1. Render markdown → HTML (mistune)
      2. Parse HTML → structured text (BeautifulSoup)
      3. Replace headings, lists, code as readable tokens
      4. Collapse whitespace and HTML entities
      5. Strip residual markdown punctuation (# * `)
      6. Truncate at a sentence boundary near limit

    If dependencies missing, fallback to naive markdown stripping.
    """
    if not md_text:
        return ""
    if not _SANITIZE_DEPS:
        # fall back to simple strip
        base = clean_markdown(md_text)
        base = _WHITESPACE_RE.sub(' ', base).strip()
        return _truncate(base, max_chars)
    try:
        html_rendered = mistune.html(md_text)  # type: ignore[attr-defined]
        soup = BeautifulSoup(html_rendered, 'html.parser')  # type: ignore
        text = _convert_html_to_text(soup)
        text = html.unescape(text)
        # Remove leftover markdown punctuation characters
        text = _MD_STRIP_PATTERN.sub('', text)
        # Normalize whitespace
        text = _WHITESPACE_RE.sub(' ', text).strip()
        text = _truncate(text, max_chars)
        # Safety: ensure no markdown structural tokens remain
        text = text.replace('#', '').replace('```', '').replace('*', '')
        return text
    except Exception:
        simple = clean_markdown(md_text)
        simple = _WHITESPACE_RE.sub(' ', simple).strip()
        return _truncate(simple, max_chars)

__all__ = [
    'clean_markdown',
    'normalize_for_tts',
]

"""Preprocessing pipeline for converting raw LLM output into clean plain English
suitable for script planning (Gemini 2.5 Pro prompt input).

Steps (spec adherence):
 1. Strip markdown symbols: # * > - _ ~ `
 2. Remove HTML/XML tags <tag> </tag>
 3. Remove special characters: " ' ; : & ^ % $ @ | \\ / { } [ ] ( )
 4. Remove escape sequences like \n, \t (normalize all whitespace to single space)
 5. Keep only letters, numbers, common punctuation (.,!?), commas, basic sentence delimiters
 6. Normalize whitespace & trim
 7. Sentence segmentation & capitalization (capitalize first letter of each sentence)
 8. Drop orphan tokens (single stray symbols) and ensure readability

Return: cleaned string (empty string if nothing meaningful remains)
"""
from __future__ import annotations
import re
from typing import List

# Markdown symbols to strip (will replace with space); keep dash handled later.
_MD_SYMS_RE = re.compile(r"[#*$*_`~>]")
# Remove markdown list dashes separately after tag strip
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPECIAL_CHARS_RE = re.compile(r"[\"';&:^%$@|\\/{}\[\]()]")
_WHITESPACE_RE = re.compile(r"\s+")
# Allow letters, numbers, space, period, comma, exclamation, question, hyphen (inside words) and basic apostrophes removed earlier
_ALLOWED_RE = re.compile(r"[^A-Za-z0-9.,!?\- ]+")
_MULTISPACE_RE = re.compile(r" {2,}")
_SENTENCE_END_RE = re.compile(r"([.!?])+")

def preprocess_for_script(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    # Replace newlines / tabs with space early
    text = text.replace("\n", " ").replace("\t", " ")
    # Strip HTML/XML tags
    text = _HTML_TAG_RE.sub(" ", text)
    # Strip markdown symbols (broad set)
    text = _MD_SYMS_RE.sub(" ", text)
    # Remove remaining list dashes only if isolated (dash surrounded by spaces)
    text = re.sub(r"\s-\s", " ", text)
    # Remove special characters
    text = _SPECIAL_CHARS_RE.sub(" ", text)
    # Collapse to allowed character set (strip other symbols)
    text = _ALLOWED_RE.sub(" ", text)
    # Normalize whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    # Sentence segmentation: split on periods/exclamation/question
    sentences: List[str] = []
    buf = []
    for token in re.split(r"([.!?])", text):
        if not token:
            continue
        if token in ".!?":
            # end of sentence
            if buf:
                sentence = "".join(buf).strip()
                if sentence:
                    sentences.append(sentence)
                buf = []
        else:
            buf.append(token)
    # Add residual buffer
    if buf:
        residual = "".join(buf).strip()
        if residual:
            sentences.append(residual)
    # Capitalize first letter per sentence; drop very short orphan tokens (<2 chars)
    cleaned: List[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 2:
            continue
        # Remove stray hyphens at ends
        s = s.strip("- ")
        if not s:
            continue
        s = s[0].upper() + s[1:]
        cleaned.append(s)
    out = ". ".join(cleaned)
    if out and not out.endswith(('.', '!', '?')):
        out += '.'
    # Final whitespace normalization
    out = _MULTISPACE_RE.sub(' ', out).strip()
    return out

__all__ = ["preprocess_for_script"]

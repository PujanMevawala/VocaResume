"""Text / markdown related helper utilities."""
import re

_MD_STRIP_PATTERN = re.compile(r"[#*_`]+")

def clean_markdown(text: str) -> str:
    """Remove simple markdown formatting tokens to display as plain text.
    This is intentionally conservative; extend as needed.
    """
    if not text:
        return ""
    return _MD_STRIP_PATTERN.sub("", text)

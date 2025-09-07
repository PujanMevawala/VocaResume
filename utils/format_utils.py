import re
from typing import List


SECTION_PATTERNS = [
    r"strengths?",
    r"weaknesses|areas? to improve",
    r"recommendations|suggestions",
    r"job fit score",
    r"interview questions?",
]


def split_sections(raw: str) -> List[str]:
    if not raw:
        return []
    # Normalize line endings
    text = raw.replace('\r','')
    # Simple heuristic: double newlines mark section breaks
    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]
    merged = []
    for c in chunks:
        if merged and len(c.split()) < 6:  # tiny fragment -> append to previous
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged


def classify_title(block: str) -> str:
    header = block.split('\n',1)[0][:80].lower()
    for pat in SECTION_PATTERNS:
        if re.search(pat, header):
            return header.title()
    # Fallback generic
    return header.title() if len(header.split()) <= 10 else "Section"


def to_structured_html(raw: str, style: str = "detailed") -> str:
    sections = split_sections(raw)
    if not sections:
        return "<div class='ta-empty'>No content.</div>"
    html_parts = []
    for idx, block in enumerate(sections, start=1):
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        title = classify_title(lines[0])
        body_lines = lines[1:] if len(lines) > 1 else []
        # Bullet detection
        bullets = []
        normal = []
        for ln in body_lines:
            if re.match(r"^[-*•]\s+", ln):
                bullets.append(re.sub(r"^[-*•]\s+", "", ln))
            elif re.match(r"^\d+[.)]\s+", ln):
                bullets.append(ln)
            else:
                normal.append(ln)
        body_html = ""
        if normal:
            body_html += f"<p>{' '.join(normal)}</p>"
        if bullets:
            tag = 'ol' if all(re.match(r"^\d+", b) for b in bullets) else 'ul'
            clean_bullets = [re.sub(r"^\d+[.)]\s+", "", b) for b in bullets]
            body_html += "<%s>" % tag + ''.join(f"<li>{re.escape(b)}</li>" for b in clean_bullets) + f"</{tag}>"
        card_class = 'compact' if style == 'compact' else 'detailed'
        html_parts.append(f"""
        <div class='ta-section-card {card_class}'>
            <div class='ta-section-head'><span class='idx'>{idx:02d}</span><h4>{title}</h4></div>
            <div class='ta-section-body'>{body_html}</div>
        </div>
        """)
    return "<div class='ta-sections-grid'>" + "".join(html_parts) + "</div>"

__all__ = ["to_structured_html"]

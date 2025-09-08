from utils.speech_adapter import generate_spoken_version, build_ssml
from utils.text_utils import clean_markdown

RAW_MD = """## Resume Analysis
**Strengths:** Python, Data Engineering, Collaboration
**Weaknesses:** Limited cloud depth
**Suggestions:** Pursue AWS certification; add deployment metrics
**Job Fit Score:** 82
Overall: Strong alignment with data pipeline roles.
"""

def test_generate_spoken_version_basic():
    spoken = generate_spoken_version(RAW_MD, user_name="John")
    assert spoken
    assert len(spoken.split()) < 300
    lowered = spoken.lower()
    assert "strength" in lowered or "great news" in lowered
    assert "weak" in lowered or "improve" in lowered
    # no markdown tokens
    assert '#' not in spoken and '*' not in spoken and '```' not in spoken


def test_build_ssml_contains_tags():
    spoken = generate_spoken_version(RAW_MD)
    ssml = build_ssml(spoken)
    assert ssml.startswith('<speak>') and ssml.endswith('</speak>')
    assert '<break' in ssml or '<emphasis>' in ssml
    # cleaned
    assert '#' not in ssml and '*' not in ssml

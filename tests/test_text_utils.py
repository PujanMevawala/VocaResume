import pytest
from utils.text_utils import normalize_for_tts

MD_CASES = [
    "# Heading 1\n## Heading 2\nSome **bold** text and *italic* text.",
    "List:\n- item one\n- item two\n\n```python\nprint('hi')\n```",
    "Mixed `code` inline and a paragraph with    extra   spaces.",
    "### Deep Heading\nContent under heading.",
]

def test_normalize_basic_no_markdown_tokens():
    for md in MD_CASES:
        out = normalize_for_tts(md)
        assert out
        assert '#' not in out
        assert '`' not in out
        assert '```' not in out
        assert '*' not in out


def test_truncation():
    long_text = "# Title\n" + ("word " * 3000)
    out = normalize_for_tts(long_text, max_chars=300)
    assert len(out) <= 310  # truncated with ellipsis
    assert out.endswith('…') or out.endswith(' ...') or out.endswith(' …')


def test_code_block_omission():
    md = "Here is code:\n```python\nprint('hello')\n```\nEnd."
    out = normalize_for_tts(md)
    # Should not include triple backticks
    assert '```' not in out

from utils.preprocess_utils import preprocess_for_script

def test_basic_markdown_and_html_strip():
    raw = "## Heading <b>Bold</b> *em* List:\n- item one\n- item two"
    out = preprocess_for_script(raw)
    assert 'Heading' in out
    assert '<b>' not in out and '*' not in out and '#' not in out


def test_special_chars_removed_and_capitalized():
    raw = "python & data engineering: strong skills!!! some orphan token @@@ next line\nnew sentence?"
    out = preprocess_for_script(raw)
    assert '&' not in out and '@' not in out
    # sentences capitalized
    parts = [p.strip() for p in out.split('.') if p.strip()]
    for p in parts:
        assert p[0].isupper()


def test_empty_result_when_no_content():
    assert preprocess_for_script("   ### $$$ !!!   ") == ""


def test_sentence_termination():
    raw = "first sentence without period second sentence? third!"
    out = preprocess_for_script(raw)
    assert out.endswith('.') or out.endswith('!') or out.endswith('?')

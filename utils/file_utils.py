"""Robust PDF ingestion utilities using Docling (primary) with fallbacks.

Returns a list of a single dict: [{"mime_type": "text/plain", "data": base64_text_or_plain}]
Where data is raw UTF-8 text (NOT base64) to keep backward compatibility with
model_service which expects a base64-like field but only slices first 200 chars.

Previous implementation generated an image of first page which broke for empty
or malformed PDFs ("Unable to get page count"). We now extract textual content
for higher fidelity downstream semantic routing + analysis.
"""
from __future__ import annotations
import streamlit as st
from io import BytesIO
import base64

@st.cache_data(show_spinner=False)
def _extract_pdf_text(bytes_data: bytes) -> str:
    # Try Docling first
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        conv = DocumentConverter()
        doc = conv.convert_bytes(bytes_data, file_type="pdf")
        text = doc.document.export_to_text() if hasattr(doc, 'document') else ''
        if text and text.strip():
            return text.strip()
    except Exception as e:  # pragma: no cover - best effort
        st.debug(f"Docling failed, falling back. {e}") if hasattr(st, 'debug') else None
    # Fallback: PyPDF2
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(BytesIO(bytes_data))
        pages = []
        for p in reader.pages[:25]:  # safety cap
            try:
                pages.append(p.extract_text() or '')
            except Exception:
                continue
        text = '\n'.join(pages).strip()
        if text:
            return text
    except Exception:
        pass
    return ''


def input_pdf_setup(uploaded_file):
    """Parse an uploaded PDF once and cache its textual content.

    Avoids multiple .read() consumption issues by storing bytes in session state.
    Returns list with one dict of mime_type text/plain.
    """
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")
    try:
        # Cache original bytes in session to prevent empty reads later
        if 'uploaded_pdf_bytes' not in st.session_state:
            st.session_state['uploaded_pdf_bytes'] = uploaded_file.read()
        data = st.session_state['uploaded_pdf_bytes']
        # Reset file pointer for any other consumer expecting to read
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        if not data or len(data) < 8:  # minimal PDF header length '%PDF-'
            st.error("Uploaded PDF appears empty or invalid (too small).")
            return None
        text_key = 'uploaded_pdf_text'
        if text_key not in st.session_state:
            text = _extract_pdf_text(data)
            st.session_state[text_key] = text
        else:
            text = st.session_state[text_key]
        if not text:
            st.error("Could not extract text from PDF (possibly scanned image-only).")
            return None
        return [{"mime_type": "text/plain", "data": text}]
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

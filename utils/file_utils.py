# PDF and file processing utilities
import base64
from io import BytesIO
import pdf2image
import streamlit as st

@st.cache_data(show_spinner=False)
def process_pdf_first_page(file_bytes: bytes):
    """Convert first page of PDF bytes to base64 JPEG payload list."""
    if not file_bytes or len(file_bytes) < 50:  # arbitrary minimal size
        raise ValueError("PDF appears empty or too small to be valid")
    try:
        images = pdf2image.convert_from_bytes(file_bytes, first_page=1, last_page=1)
    except Exception as e:
        # Provide clearer diagnostic for empty/corrupt PDFs
        raise RuntimeError(f"Unable to read first page: {e}") from e
    first_page = images[0]
    buf = BytesIO()
    first_page.save(buf, format='JPEG')
    return [{"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}]

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            if not file_bytes:
                raise ValueError("Uploaded PDF file is empty")
            return process_pdf_first_page(file_bytes)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    else:
        raise FileNotFoundError("No file uploaded")

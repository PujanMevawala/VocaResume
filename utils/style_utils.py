import streamlit as st
import os
from pathlib import Path

def load_css():
    """
    Load CSS for the Streamlit app.
    Attempts to load from external file first, falls back to inline if needed.
    """
    # Get the base directory
    base_dir = Path(__file__).parent.parent

    # Try to load the external CSS file
    css_path = base_dir / "static" / "style.css"
    try:
        if css_path.exists():
            with open(css_path) as f:
                css_content = f.read()
                st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
                print(f"Successfully loaded CSS from {css_path}")
                return True
        else:
            # If we couldn't find the file, log the paths for debugging
            print(f"CSS file not found at: {css_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            # Fall back to loading the CSS from the file directly in the app.py location
            alt_path = Path(os.getcwd()) / "static" / "style.css"
            if alt_path.exists():
                with open(alt_path) as f:
                    css_content = f.read()
                    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
                    print(f"Successfully loaded CSS from alternate path: {alt_path}")
                    return True
            
            # If everything fails, load a minimal inline style to ensure proper display
            _load_minimal_inline_css()
            return False
    except Exception as e:
        print(f"Error loading CSS: {str(e)}")
        _load_minimal_inline_css()
        return False

def _load_minimal_inline_css():
    """Load a minimal inline CSS as fallback to ensure proper display"""
    minimal_css = """
    /* Minimal fallback CSS */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-color: #0088a9;
        --secondary-color: #e67e22;
    }
    
    .ta-logo-wrapper { 
        text-align: center; 
        margin-bottom: 1rem;
        padding-top: 1.2rem;
    }
    
    .ta-logo-wrapper img { 
        max-width: 60px;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        border: 2px solid rgba(0, 136, 169, 0.1);
    }
    
    .ta-brand-title { 
        font-size: 1.2rem; 
        font-weight: 700; 
        color: #0088a9; 
        margin-top: 0.5rem; 
    }
    
    .ta-tagline { 
        font-size: 0.8rem; 
        letter-spacing: 0.5px; 
        color: #5d7182; 
        margin-bottom: 1rem; 
    }
    
    .ta-side-section {
        margin-bottom: 1.2rem;
        padding: 1rem;
        border: none;
        border-radius: 12px;
        background: #f0f9ff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    
    .ta-side-section h4 {
        margin: 0 0 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #0088a9;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .ta-footer { 
        text-align: center; 
        font-size: 0.75rem; 
        padding: 1rem 0.5rem; 
        color: #5d7182; 
        opacity: 0.8; 
    }
    """
    st.markdown(f"<style>{minimal_css}</style>", unsafe_allow_html=True)
    print("Loaded minimal inline CSS as fallback")

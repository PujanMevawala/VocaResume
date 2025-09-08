"""Main Streamlit application for VocaResume (refactored).

This module provides the streamlined resume analysis + optional TTS narration UI.
Speech-to-text recording has been fully removed. Voice toggle now only controls
automatic TTS playback of generated summaries.
"""

from __future__ import annotations

# Standard libs
import os, re, hashlib

import streamlit as st

# Fast test short-circuit before expensive imports
SHORT_TEST_MODE = os.getenv("SHORT_TEST_MODE") == "1"
if SHORT_TEST_MODE:
    st.write("Test mode active – quick import path.")
    # Provide minimal placeholders required by tests
    class _Dummy:
        AVAILABLE_MODELS = {"Gemini 2.5 Pro": {"provider": "google", "model": "gemini-2.5-pro"}}
        GROQ_API_KEY = GOOGLE_API_KEY = PPLX_API_KEY = None
    settings = _Dummy()
    st.stop()

import google.generativeai as genai
from groq import Groq  # type: ignore
from openai import OpenAI  # Perplexity-compatible
import plotly.graph_objects as go

# Internal modules
# Settings module exports constants (GROQ_API_KEY, AVAILABLE_MODELS, etc.)
try:
    from services import settings as settings  # preferred
except Exception:
    try:
        from config import settings as settings  # legacy
    except Exception:
        class _FallbackSettings:
            GROQ_API_KEY = None
            GOOGLE_API_KEY = None
            PPLX_API_KEY = None
            AVAILABLE_MODELS = {"Gemini 2.5 Pro": {"provider": "google", "model": "gemini-2.5-pro"}}
        settings = _FallbackSettings()  # type: ignore
from services.model_service import get_model_response
from services.voice_script_service import generate_voice_script
from tasks.task_factory import create_tasks, get_task_from_query
from agents.agent_factory import create_agents
from utils.voice_utils import text_to_speech, voice_enabled, voice_stack_report
from utils.file_utils import input_pdf_setup
from utils.text_utils import clean_markdown


def _init_session_state():
    st.session_state.setdefault('app_view', 'landing')
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('responses', {})
    st.session_state.setdefault('voice_played', {})


def _render_landing_page():
    st.markdown(
        """
        <div class='landing-container'>
            <div class='landing-hero'>
                <h1>VocaResume</h1>
                <p class='subtitle'>Modern career insights with streamlined voice-ready narration.</p>
            </div>
            <div style='margin-top:2rem;text-align:center;'>
                <a href='?view=main' style='text-decoration:none;'>
                    <button style='padding:.75rem 1.5rem;border:none;background:#0b7285;color:#fff;border-radius:8px;font-size:1rem;cursor:pointer;'>Enter App</button>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Enter Application", key="enter_app_btn"):
        st.session_state['app_view'] = 'main'
        st.rerun()


_init_session_state()

if st.session_state.get('app_view') == 'landing':
    _render_landing_page()
    st.stop()

# Initialize API clients with error handling
if settings.GROQ_API_KEY:
    groq_client = Groq(api_key=settings.GROQ_API_KEY)
else:
    groq_client = None
    st.warning("⚠️ GROQ_API_KEY not found. Some features may not work.")

if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY not found. Some features may not work.")

# Perplexity (OpenAI-compatible) client
if settings.PPLX_API_KEY:
    try:
        pplx_client = OpenAI(base_url="https://api.perplexity.ai", api_key=settings.PPLX_API_KEY)
    except Exception:
        pplx_client = None
        st.warning("⚠️ Failed to initialize Perplexity client.")
else:
    pplx_client = None
    st.info("ℹ️ PPLX_API_KEY not found. Perplexity models will be unavailable.")


# (Welcome card removed for cleaner layout; hero now only on landing page)


######## Redesigned Single-Page Layout ########

# Sticky Glass Header (logo)
brand_svg = """
<svg viewBox='0 0 64 64' width='64' height='64' role='img' aria-label='App Logo'>
  <defs>
    <linearGradient id='brandGrad' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#0088a9'/>
      <stop offset='100%' stop-color='#e67e22'/>
    </linearGradient>
  </defs>
  <circle cx='32' cy='32' r='28' fill='url(#brandGrad)' opacity='0.15'></circle>
  <circle cx='32' cy='32' r='20' fill='none' stroke='url(#brandGrad)' stroke-width='6' stroke-linecap='round' class='brand-ring'></circle>
  <circle cx='32' cy='32' r='6' fill='url(#brandGrad)'></circle>
</svg>
"""
st.markdown("<div class='ta-header' id='top'>", unsafe_allow_html=True)
header_cols = st.columns([0.9, 6])
with header_cols[0]:
    st.markdown(f"<div class='brand-anim-logo'>{brand_svg}</div>", unsafe_allow_html=True)
with header_cols[1]:
    st.markdown("<div class='brand-wrap'><span class='brand-title'>VocaResume</span><div class='brand-sub'>Voice-Powered Career Copilot</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Control Panel (structured)
st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
cp_cols_top = st.columns([2.6, 1, 1])
with cp_cols_top[0]:
    model_choice = st.selectbox("Model", list(settings.AVAILABLE_MODELS.keys()), index=0, key="model_select")
    model_info = settings.AVAILABLE_MODELS[model_choice]
    provider = model_info.get('provider')
    model = model_info.get('model')
    st.caption(f"Provider: {provider.title() if provider else 'None'} | Model: {model if model else 'None'}")
    if not provider or not model:
        st.error(f"Model config error: provider or model is missing (provider={provider}, model={model})")
    elif provider not in settings.AVAILABLE_MODELS[model_choice]['provider']:
        st.warning(f"Provider '{provider}' not recognized. Check AVAILABLE_MODELS in config.")
with cp_cols_top[1]:
    voice_mode = st.toggle("Voice", value=False, help="Enable speech input & spoken responses.")
    st.markdown(f"<div class='voice-anim {'on' if voice_mode else 'off'}'><span></span><span></span><span></span></div>", unsafe_allow_html=True)
with cp_cols_top[2]:
    if st.button("Clear Cache", type="secondary"):
        if 'responses' in st.session_state:
            st.session_state['responses'].clear()
            st.toast("Cache Cleared!")

max_tokens = st.slider("Max Tokens", 512, 8192, 4096, 256, key="max_tokens_slider")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='divider-fw'></div>", unsafe_allow_html=True)

st.markdown("""
<style>
/* Modern minimalist layout - widened */
.layout-grid {display:grid;grid-template-columns:500px 1fr;gap:2.25rem;align-items:start;margin-top:1.25rem;max-width:1900px;margin-left:auto;margin-right:auto;}
@media (max-width:1400px){.layout-grid{grid-template-columns:460px 1fr;}}
@media (max-width:1250px){.layout-grid{grid-template-columns:420px 1fr;}}
@media (max-width:1150px){.layout-grid{grid-template-columns:1fr;}}
.panel {background:rgba(255,255,255,0.60);backdrop-filter:blur(18px);border:1px solid rgba(0,0,0,.06);border-radius:22px;padding:1.15rem 1.35rem 1.35rem;box-shadow:0 6px 22px -6px rgba(0,0,0,.08);} 
.panel h3 {margin:0 0 .85rem;font-size:1.08rem;font-weight:600;letter-spacing:.5px;display:flex;align-items:center;gap:.5rem;}
.panel small{opacity:.75;}
.btn-primary {background:#0b7285;color:#fff;border:none;padding:.65rem 1.1rem;border-radius:10px;font-weight:500;font-size:.9rem;cursor:pointer;display:inline-flex;align-items:center;gap:.5rem;}
.status-tag {display:inline-block;padding:.3rem .65rem;border-radius:999px;font-size:.60rem;letter-spacing:.65px;text-transform:uppercase;background:#0b7285;color:#fff;margin-left:.5rem;}
.output-block {background:#0d1117;color:#d0d4da;font-family:var(--font-mono, ui-monospace, Menlo, monospace);padding:1.1rem 1.15rem;border-radius:18px;font-size:.8rem;line-height:1.5;position:relative;border:1px solid #1d2630;}
.output-block h4{margin:.25rem 0 .65rem;font-size:.75rem;text-transform:uppercase;letter-spacing:.85px;font-weight:600;color:#6ec9dc;}
.pill {display:inline-flex;align-items:center;font-size:.62rem;padding:.25rem .65rem;border-radius:18px;letter-spacing:.65px;background:#eef8f9;color:#0b7285;margin:0 .4rem .4rem 0;font-weight:600;}
.audio-bar {position:sticky;bottom:0;left:0;width:100%;background:linear-gradient(180deg,rgba(255,255,255,.15),rgba(255,255,255,.55));backdrop-filter:blur(14px);border-top:1px solid rgba(0,0,0,.08);padding:.6rem 1rem;margin-top:2rem;border-radius:14px;}
button[disabled]{opacity:.55 !important;cursor:not-allowed !important;}
</style>
""", unsafe_allow_html=True)

# New Two-Column Layout
col_wrap = st.container()
with col_wrap:
    st.markdown("<div class='layout-grid'>", unsafe_allow_html=True)
    # LEFT PANEL: Inputs
    st.markdown("<div class='panel' id='left-panel'>", unsafe_allow_html=True)
    st.markdown("<h3>Input <span class='status-tag'>SOURCE</span></h3>", unsafe_allow_html=True)
    input_text = st.text_area("Job Description", key="input", placeholder="Paste the job description...", height=140)
    uploaded_file = st.file_uploader("Resume (PDF)", type=["pdf"], help="Optional – improves relevance.")
    user_query = st.text_input("Your Query", key="user_query", placeholder="Ask e.g. 'Give me interview questions' or 'What's my fit score?' or just 'Analyze resume'")
    user_name = st.text_input("Name (personalize voice)", key="user_name", placeholder="Optional")
    auto_task = None
    detected_task_label = None
    if st.button("Generate", type="primary", use_container_width=True, disabled=not (input_text and uploaded_file and user_query)):
        st.session_state['last_run'] = {
            'input_text': input_text,
            'user_query': user_query
        }
        # classify task
        t_idx, intent = get_task_from_query(user_query)
        detected_task_label = intent
        auto_task = t_idx
        st.session_state['detected_task'] = {'index': t_idx, 'intent': intent}
    st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT PANEL: Outputs
    st.markdown("<div class='panel' id='right-panel'>", unsafe_allow_html=True)
    st.markdown("<h3>Output <span class='status-tag'>RESULT</span></h3>", unsafe_allow_html=True)
    # If run executed, perform pipeline
    if 'detected_task' in st.session_state and st.session_state.get('last_run'):
        if not uploaded_file:
            st.warning("Upload a resume PDF to proceed.")
        else:
            # Lazy agent/task creation
            if 'cached_tasks' not in st.session_state:
                agents = create_agents(settings.AVAILABLE_MODELS[model_choice])
                st.session_state['cached_agents'] = agents
                st.session_state['cached_tasks'] = create_tasks(agents)
            tasks_local = st.session_state['cached_tasks']
            tinfo = st.session_state['detected_task']
            t_idx = tinfo['index']
            intent = tinfo['intent']
            # Load resume
            try:
                pdf_content = input_pdf_setup(uploaded_file)
            except Exception as e:
                st.error(f"Resume parse failed: {e}")
                pdf_content = None
            if pdf_content:
                with st.spinner(f"Running task: {intent.replace('_',' ').title()}..."):
                    result = get_model_response(input_text, pdf_content, tasks_local[t_idx].description, settings.AVAILABLE_MODELS[model_choice], groq_client, pplx_client, max_output_tokens=max_tokens)
                raw_md = result.get('display_md','') if isinstance(result, dict) else str(result)
                from utils.preprocess_utils import preprocess_for_script
                cleaned = preprocess_for_script(raw_md)
                from services.script_planner import plan_script
                script_plan = plan_script(cleaned, settings.GOOGLE_API_KEY)
                if script_plan.get('status') != 'success':
                    st.error(f"Script planning failed: {script_plan.get('message')}")
                else:
                    st.session_state['last_pipeline'] = {
                        'task': intent,
                        'raw': raw_md,
                        'cleaned': cleaned,
                        'plan': script_plan.get('plan',''),
                        'script': script_plan.get('script','')
                    }
    pipe = st.session_state.get('last_pipeline')
    if pipe:
        st.markdown(f"<div class='pill'>Task: {pipe['task']}</div>", unsafe_allow_html=True)
        with st.expander("Result", expanded=True):
            st.markdown(f"<div class='output-block'><h4>PRIMARY OUTPUT</h4>{clean_markdown(pipe['raw'])}</div>", unsafe_allow_html=True)
        # Script output intentionally hidden from UI to reduce clutter per user request
        if voice_mode and pipe.get('script'):
            audio_bytes = text_to_speech(pipe['script'][:1800])  # still use improved script internally
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mpeg')
                st.download_button("Download Narration", data=audio_bytes, file_name="vocaresume_narration.mp3", mime="audio/mpeg", use_container_width=True)
    else:
        st.info("Enter inputs and click Generate to see results.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Progress bar removed in streamlined UI

# No voice recording (STT removed) – voice_mode defined later in control panel

# Diagnostics (optional) – show when VOICE_DEBUG enabled
if os.getenv("VOICE_DEBUG", "").lower() in {"1","true","yes"}:
    with st.expander("Voice Diagnostics"):
        st.json(voice_stack_report())

progress_bar = st.progress(0, text="Idle")

# Ensure session caches
if 'responses' not in st.session_state:
    st.session_state['responses'] = {}
if 'voice_played' not in st.session_state:
    st.session_state['voice_played'] = {}
if 'history' not in st.session_state:
    st.session_state['history'] = []


def cache_and_return(key, compute_fn):
    if key in st.session_state['responses']:
        return st.session_state['responses'][key]
    result = compute_fn()
    if result:
        st.session_state['responses'][key] = result
    return result

# Load resume content if uploaded
pdf_content = None
if uploaded_file is not None:
    try:
        pdf_content = input_pdf_setup(uploaded_file)
    except Exception:
        st.error("Failed to process the PDF.")

# Initialize agents & tasks only when resume provided to save time
agents = tasks = None
if pdf_content:
    agents = create_agents(model_info)
    tasks = create_tasks(agents)

def run_task(task_index: int, cache_key: tuple[str,str]):
    """Generic runner for tasks with progress feedback.

    Returns a dict with keys: display_md, tts_text.
    """
    if not tasks:
        st.warning("Upload a resume first.")
        return {"display_md": "", "tts_text": ""}
    def _compute():
        labels = {0:"Analysis",1:"Interview",2:"Suggestions",3:"Job Fit"}
        label = labels.get(task_index, "Task")
        progress_bar.progress(0, text=f"Starting {label}...")
        with st.spinner(f"Running {label} task..."):
            progress_bar.progress(25, text="Sending to model...")
            r = get_model_response(input_text, pdf_content, tasks[task_index].description, model_info, groq_client, pplx_client, max_output_tokens=max_tokens)
            progress_bar.progress(100, text=f"{label} Complete!")
            # Backward safety: if legacy string returned somehow
            if isinstance(r, str):
                return {"display_md": r, "tts_text": clean_markdown(r)}
            return r
    return cache_and_return(cache_key, _compute)



# Sections

# Floating Mic HTML injection
# Floating mic removed (STT deprecated)

with st.expander("Session History"):
    for i,h in enumerate(st.session_state.get('history', [])[-30:]):
        st.markdown(f"**{i+1}. {h['type']} :: {h.get('section','')}**")
        st.markdown(f"<div style='white-space:pre-line;font-size:.8rem;'>{clean_markdown(h['text'])}</div>", unsafe_allow_html=True)
        if h.get('audio_bytes'):
            st.audio(h['audio_bytes'], format='audio/wav')

st.markdown("<div style='margin-top:3rem;text-align:center;opacity:.55;font-size:.7rem;'>VocaResume – Modern Interface</div>", unsafe_allow_html=True)
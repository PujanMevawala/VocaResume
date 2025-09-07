import warnings
import logging
"""VocaResume Streamlit App."""
from dotenv import load_dotenv
import base64
import streamlit as st

# MUST be first Streamlit command; guard against re-entry in certain testing contexts
if not getattr(st.runtime, "_is_running_with_streamlit", False):  # attribute may not exist
    try:
        st.set_page_config(
            page_title="VocaResume",
            page_icon="�",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

import os
# Early test fast-exit BEFORE heavy imports
if os.getenv("SHORT_TEST_MODE", "").lower() in {"1", "true", "yes"} or os.getenv("PYTEST_CURRENT_TEST"):
    st.title("VocaResume (Test Mode)")
    st.write("Test mode active - skipping heavy initialization.")
    st.stop()

from groq import Groq
import google.generativeai as genai
from utils.style_utils import load_css
import plotly.graph_objects as go
import re
from openai import OpenAI
import hashlib
from services.model_service import get_model_response
from utils.file_utils import input_pdf_setup
from services.agent_factory import create_agents
from services.task_factory import create_tasks, get_task_from_query
from utils.voice_utils import record_audio, speech_to_text, text_to_speech, voice_enabled, voice_stack_report
from services import settings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"pydub\.utils"
)
load_dotenv()
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DEBUG"] = "false"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
load_css()

# ---------------- Landing Page Routing ---------------- #
if 'app_view' not in st.session_state:
    st.session_state['app_view'] = 'landing'

# Support navigation via query parameter for pure HTML CTA using new st.query_params API
try:
    view_param = st.query_params.get("view")
    # st.query_params returns str (new API) but handle list for safety if legacy object shape appears
    if view_param == 'main' or (isinstance(view_param, list) and view_param and view_param[0] == 'main'):
        st.session_state['app_view'] = 'main'
except Exception:
    pass


def _encode_b64(path: str) -> str:
    """Read file and return base64 string (empty if missing)."""
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        return ""


def _render_landing_page():
    # Landing hero
    st.markdown(
        """
        <div class="landing-container">
            <div class="landing-hero">
                <h1>Unlock Your Career Potential</h1>
                <p class="subtitle">Analyze resumes, prepare for interviews, and assess job fit with an intelligent career copilot.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CTA Section (moved above Core Features) – single functional Streamlit button centered
    st.markdown(
        """
        <div class=\"cta-band\">\n            <div class=\"cta-copy\">\n                <h3 style='margin-bottom:.35rem;'>Ready to analyze your resume?</h3>\n                <p style='max-width:620px;margin:0 auto 1.1rem;line-height:1.5;'>Upload your PDF and paste a target job description to get instant insights.</p>\n            </div>\n        </div>\n        """,
        unsafe_allow_html=True,
    )

    # Centered button directly under text
    btn_center_cols = st.columns([3,4,3])
    with btn_center_cols[1]:
        if st.button("Start Your Journey", key="landing_start_top", help="Go to main app", use_container_width=True):
            st.session_state["app_view"] = "main"
            st.rerun()

    # Core Features
    st.markdown(
        """
        <div class="landing-section">
            <h2>Core Features</h2>
            <div class="features-grid">
                <div class="feature-card" style="animation-delay:.05s;">
                    <div class="abstract-glyph glyph-a"></div>
                    <h3>Resume Analysis</h3>
                    <p>Balanced strengths & weaknesses versus any job description.</p>
                </div>
                <div class="feature-card" style="animation-delay:.10s;">
                    <div class="abstract-glyph glyph-b"></div>
                    <h3>Job Fit Score</h3>
                    <p>Quantitative compatibility scoring plus qualitative explanation.</p>
                </div>
                <div class="feature-card" style="animation-delay:.15s;">
                    <div class="abstract-glyph glyph-c"></div>
                    <h3>Interview Prep</h3>
                    <p>Tailored technical questions derived from your profile context.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # (CTA moved above features)

    # How It Works
    st.markdown(
        """
        <div class="landing-section">
            <h2>How It Works</h2>
            <div class="how-grid">
                <div class="how-card">
                    <div class="how-object obj-1"></div>
                    <div class="how-content">
                        <h3>Provide Inputs</h3>
                        <p>Paste a job description & upload a PDF resume.</p>
                    </div>
                </div>
                <div class="how-card">
                    <div class="how-object obj-2"></div>
                    <div class="how-content">
                        <h3>Select Action</h3>
                        <p>Choose analysis, suggestions, interview prep, or fit scoring.</p>
                    </div>
                </div>
                <div class="how-card">
                    <div class="how-object obj-3"></div>
                    <div class="how-content">
                        <h3>Get Insights</h3>
                        <p>Downloadable & voice-playable AI generated output.</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state['app_view'] == 'landing':
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
    if st.button("Play Test Voice"):
        sample = "Voice is configured successfully. This is a test message."
        audio_out = text_to_speech(sample)
        if audio_out:
            st.audio(audio_out, format='audio/wav')
        else:
            st.error("TTS failed to produce audio. Check pyttsx3 and ffmpeg.")

max_tokens = st.slider("Max Tokens", 512, 8192, 4096, 256, key="max_tokens_slider")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='divider-fw'></div>", unsafe_allow_html=True)

# Anchor Chip Navigation
anchor_html = """
<div class='anchor-chips-wrapper'>
    <div class='anchor-chips'>
        <a class='chip' href='#inputs'>INPUTS</a>
        <a class='chip' href='#analysis'>ANALYSIS</a>
        <a class='chip' href='#suggestions'>SUGGESTIONS</a>
        <a class='chip' href='#interview'>INTERVIEW</a>
        <a class='chip' href='#jobfit'>JOB FIT</a>
    </div>
</div>
"""
st.markdown(anchor_html, unsafe_allow_html=True)



# Input Section
st.markdown('<span id="inputs" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("<div class='ta-section-card' id='card-inputs'>", unsafe_allow_html=True)
st.markdown("<h2>Inputs</h2><div class='divider-fw'></div>", unsafe_allow_html=True)
input_cols = st.columns([2.2,1])
with input_cols[0]:
    input_text = st.text_area("Job Description", key="input", placeholder="Paste or dictate the job description...", height=160)
with input_cols[1]:
    uploaded_file = st.file_uploader("Resume (PDF)", type=["pdf"])
    st.caption("Provide a recent tailored resume PDF.")
st.markdown("</div>", unsafe_allow_html=True)

# Voice Recording & Query Routing
voice_query_result = None

# Floating mic (HTML inserted at end of page; state captured here)
if voice_mode and voice_enabled():
    st.session_state.setdefault('recording_state', False)
    mic_container = st.empty()
    # Provide inline recorder area hidden to user (logic only) when triggered by JS toggling a hidden button
    # Hidden toggle button (wrapped for styling / hiding)
    st.markdown("<div class='hidden-mic-btn-wrap'>", unsafe_allow_html=True)
    if st.button("Voice Record Toggle", key="hidden_mic_button", help="Hidden control for mic (triggered by floating button)"):
        st.session_state['recording_state'] = not st.session_state['recording_state']
    st.markdown("</div>", unsafe_allow_html=True)
    live_transcript_placeholder = st.empty()
    if st.session_state['recording_state']:
        st.markdown("<div class='vr-native-wrap'>", unsafe_allow_html=True)
        audio_bytes = record_audio(label="Recording...", instructions="Recording active. Click to stop.")
        st.markdown("</div>", unsafe_allow_html=True)
        live_transcript_placeholder.info("Listening... release to transcribe")
        if audio_bytes:
            res = speech_to_text(audio_bytes)
            if res and res.text:
                live_transcript_placeholder.success(res.text)
                voice_query_result = res.text
                st.session_state['history'].append({"type":"voice_input","section":"raw","text":res.text})
            else:
                live_transcript_placeholder.warning("No speech detected.")
            st.session_state['recording_state'] = False

    # Advanced recorder UI (always render when voice mode on and libs available)
    rec_state = 'recording' if st.session_state.get('recording_state') else 'idle'
    status_text = 'Recording… Tap to stop' if rec_state == 'recording' else 'Idle – Tap to speak'
    mic_svg = """<svg viewBox='0 0 24 24' width='30' height='30' aria-hidden='true'><path fill='currentColor' d='M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 14 0h-2Zm-5 7a7.97 7.97 0 0 0 5-1.7v2.34A9.94 9.94 0 0 1 12 21a9.94 9.94 0 0 1-5-.36V18.3A7.97 7.97 0 0 0 12 19Z'/></svg>"""
    stop_svg = """<svg viewBox='0 0 24 24' width='30' height='30' aria-hidden='true'><rect x='6' y='6' width='12' height='12' rx='2' fill='currentColor'/></svg>"""
    icon_svg = stop_svg if rec_state == 'recording' else mic_svg
    vr_html = f"""
    <div class='voice-recorder {rec_state}' data-state='{rec_state}' id='voice-recorder' role='group' aria-label='Voice recorder module'>
        <div class='vr-btn-wrap'>
            <div class='vr-pulse'></div>
            <button type='button' class='vr-btn' id='vr-btn' aria-pressed='{str(rec_state=='recording').lower()}' aria-label='Voice recorder {rec_state} mode toggle'>
                <span class='vr-icon'>{icon_svg}</span>
            </button>
            <div class='vr-wave' aria-hidden='true'>
                <span></span><span></span><span></span><span></span><span></span>
            </div>
        </div>
        <div class='vr-meta'>
            <div class='vr-status' role='status' aria-live='polite'>{status_text}</div>
            <div class='vr-row'>
                <div class='vr-timer' id='vr-timer'>00:00</div>
                <div class='vr-hint'>Your speech is transcribed & routed intelligently.</div>
            </div>
        </div>
    </div>
    <script>
      (function(){{
         const REC_STATE = '{rec_state}';
         const doc = window.parent.document;
         const btn = doc.getElementById('vr-btn');
         const timerEl = doc.getElementById('vr-timer');
         let intervalId = null;
         function fmt(t){{const m=String(Math.floor(t/60)).padStart(2,'0');const s=String(t%60).padStart(2,'0');return m+':'+s;}}
         function startTimer(){{ if(!timerEl) return; const start=Date.now(); intervalId=setInterval(()=>{{ const diff=Math.floor((Date.now()-start)/1000); timerEl.textContent=fmt(diff); }},1000); }}
         function stopTimer(){{ if(intervalId){{clearInterval(intervalId); intervalId=null;}} if(timerEl) timerEl.textContent='00:00'; }}
         if(REC_STATE==='recording'){{ startTimer(); }} else {{ stopTimer(); }}
         function triggerHidden(){{
            const buttons = Array.from(doc.querySelectorAll('button'));
            const hidden = buttons.find(b => b.innerText.trim() === 'Voice Record Toggle');
            if(hidden){{ hidden.click(); }}
         }}
         if(btn){{ btn.addEventListener('click', triggerHidden); }}
         const hidden = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.trim() === 'Voice Record Toggle');
         if(hidden){{ hidden.style.display='none'; }}
      }})();
    </script>
    """
    st.markdown(vr_html, unsafe_allow_html=True)
elif voice_mode and not voice_enabled():
    st.info("Voice dependencies missing (vosk/pyttsx3/audiorecorder/pydub).")

# Diagnostics expander when Voice Debug enabled
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

from utils.text_utils import clean_markdown

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
    """Generic runner for tasks with progress feedback."""
    if not tasks:
        st.warning("Upload a resume first.")
        return ""
    def _compute():
        labels = {0:"Analysis",1:"Interview",2:"Suggestions",3:"Job Fit"}
        label = labels.get(task_index, "Task")
        progress_bar.progress(0, text=f"Starting {label}...")
        with st.spinner(f"Running {label} task..."):
            progress_bar.progress(25, text="Sending to model...")
            r = get_model_response(input_text, pdf_content, tasks[task_index].description, model_info, groq_client, pplx_client, max_output_tokens=max_tokens)
            progress_bar.progress(100, text=f"{label} Complete!")
            return r
    return cache_and_return(cache_key, _compute)

# If we have a voice query, auto route and execute
if voice_query_result and pdf_content and tasks:
    idx, intent = get_task_from_query(voice_query_result)
    routed_response = run_task(idx, (intent, model_choice))
    if routed_response:
        st.success(f"Voice command executed: {intent.replace('_',' ').title()}")
        st.markdown(f"### Voice Result ({intent.title()})")
        st.markdown(f'<div class="ui-card"><div style="white-space: pre-line; line-height: 1.6;">{clean_markdown(routed_response)}</div></div>', unsafe_allow_html=True)
        st.session_state['history'].append({"type":"response","section":intent,"text":routed_response})
        resp_hash = hashlib.md5(routed_response.encode('utf-8')).hexdigest()
        vp_key = (intent, model_choice, resp_hash)
        if voice_mode and vp_key not in st.session_state['voice_played']:
            audio_out = text_to_speech(routed_response[:1200], format='wav')
            if audio_out:
                st.audio(audio_out, format='audio/wav')
                st.session_state['voice_played'][vp_key] = True
                st.session_state['history'].append({"type":"tts","section":intent,"text":routed_response[:1200],"audio_bytes":audio_out})

main_sections = [
        {
            "id": "analysis",
            "title": "Resume Analysis",
            "desc": "Get a comprehensive breakdown of your resume's strengths and weaknesses versus the job description.",
            "button": "Analyze Resume",
            "task_index": 0,
            "icon": "<svg width='32' height='32'><circle cx='16' cy='16' r='14' fill='url(#grad1)'/><defs><linearGradient id='grad1' x1='0' y1='0' x2='32' y2='32'><stop offset='0%' stop-color='#0088a9'/><stop offset='100%' stop-color='#e67e22'/></linearGradient></defs></svg>"
        },
        {
            "id": "suggestions",
            "title": "Suggestions",
            "desc": "Get actionable advice to tune your resume, enhancing its appeal to both ATS and human reviewers.",
            "button": "Get Improvement Suggestions",
            "task_index": 2,
            "icon": "<svg width='32' height='32'><rect x='4' y='4' width='24' height='24' rx='8' fill='url(#grad2)'/><defs><linearGradient id='grad2' x1='0' y1='0' x2='32' y2='32'><stop offset='0%' stop-color='#e67e22'/><stop offset='100%' stop-color='#0088a9'/></linearGradient></defs></svg>"
        },
        {
            "id": "interview",
            "title": "Interview Prep",
            "desc": "Generate tailored technical questions based on your resume and the job role to help you prepare.",
            "button": "Generate 5 Technical Questions",
            "task_index": 1,
            "icon": "<svg width='32' height='32'><ellipse cx='16' cy='16' rx='14' ry='10' fill='url(#grad3)'/><defs><linearGradient id='grad3' x1='0' y1='0' x2='32' y2='32'><stop offset='0%' stop-color='#0088a9'/><stop offset='100%' stop-color='#e67e22'/></linearGradient></defs></svg>"
        },
        {
            "id": "jobfit",
            "title": "Job Fit Score",
            "desc": "Receive a quantitative score and qualitative analysis of how well your profile aligns with the job.",
            "button": "Calculate Job Fit Score",
            "task_index": 3,
            "icon": "<svg width='32' height='32'><polygon points='16,4 28,28 4,28' fill='url(#grad4)'/><defs><linearGradient id='grad4' x1='0' y1='0' x2='32' y2='32'><stop offset='0%' stop-color='#e67e22'/><stop offset='100%' stop-color='#0088a9'/></linearGradient></defs></svg>"
        }
    ]

requirements_ready = bool(input_text and input_text.strip() and pdf_content)
if not requirements_ready:
        st.info("Upload a resume PDF and enter the job description to enable actions.")

for i, sec in enumerate(main_sections):
    if i > 0:
        st.markdown("<div class='divider-fw'></div>", unsafe_allow_html=True)
    st.markdown(f"<span id='{sec['id']}' class='section-anchor'></span>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='main-section-card' id='card-{sec['id']}'>
        <div class='msc-header'>
            <div class='msc-icon'>{sec['icon']}</div>
            <div class='msc-title'>{sec['title']}</div>
        </div>
        <div class='msc-desc'>{sec['desc']}</div>
    </div>
    """, unsafe_allow_html=True)
    btn_label = sec['button'] if requirements_ready else sec['button']
    if st.button(btn_label, key=f"btn_{sec['id']}", disabled=not requirements_ready):
        run_task(sec['task_index'], (sec['id'], model_choice))
    resp = st.session_state['responses'].get((sec['id'], model_choice), "")
    if resp:
        if sec['id'] == 'analysis':
            strengths = resp.lower().count('strength')
            weaknesses = resp.lower().count('weakness')
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Strength Mentions", strengths)
            with c2:
                st.metric("Weakness Mentions", weaknesses)
        if sec['id'] == 'jobfit':
            score_match = re.search(r"Job Fit Score:\s*(\d+)", resp, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 50
            colA, colB = st.columns([1,2])
            with colA:
                st.metric("Overall Fit Score", f"{score}/100")
                if score >= 85:
                    st.success("Excellent Match")
                elif score >= 70:
                    st.info("Strong Match")
                elif score >= 50:
                    st.warning("Moderate Match")
                else:
                    st.error("Needs Improvement")
                st.caption("Score is an AI-driven estimate.")
            with colB:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={'text': "Compatibility", 'font': {'size': 18}},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "var(--primary-color)"},
                           'steps': [
                               {'range': [0, 50], 'color': "#ffe0b2"},
                               {'range': [50, 80], 'color': "#b2dfdb"},
                               {'range': [80, 100], 'color': "#80cbc4"}]}
                ))
                fig.update_layout(height=210, margin=dict(l=15, r=15, t=30, b=10), font_family="Poppins")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div class='msc-output'><div style='white-space: pre-line; line-height: 1.6;'>{clean_markdown(resp)}</div></div>", unsafe_allow_html=True)
        download_name = {
            'analysis': 'resume_analysis.txt',
            'interview': 'technical_interview_questions.txt',
            'suggestions': 'resume_suggestions.txt',
            'jobfit': 'job_fit_report.txt'
        }.get(sec['id'], 'output.txt')
        st.download_button("Download", data=resp, file_name=download_name, mime="text/plain", use_container_width=True)
        resp_hash = hashlib.md5(resp.encode('utf-8')).hexdigest()
        vp_key = (sec['id'], model_choice, resp_hash)
        # Always generate and show audio (WAV) and MP3 for every response
        # Clean markdown for TTS/MP3 so it doesn't read tags
        tts_text = clean_markdown(resp[:1200])
        audio_out = text_to_speech(tts_text, format='wav')
        if audio_out:
            st.audio(audio_out, format='audio/wav')
            if vp_key not in st.session_state['voice_played']:
                st.session_state['voice_played'][vp_key] = True
                st.session_state['history'].append({"type":"tts","section":sec['id'],"text":tts_text,"audio_bytes":audio_out})
        mp3_text = clean_markdown(resp[:1000])
        mp3_audio = text_to_speech(mp3_text, format='mp3')
        if mp3_audio:
            st.download_button("Download MP3 Audio", data=mp3_audio, file_name=sec['id']+".mp3", mime="audio/mpeg", use_container_width=True)
        # Add history of textual response (once)
        st.session_state['history'].append({"type":"response","section":sec['id'],"text":resp})

# Sections

# Floating Mic HTML injection
if voice_mode:
    mic_state_class = 'recording' if st.session_state.get('recording_state') else ''
    mic_html = """
    <div class='fab-mic {STATE}' id='fab-mic' role='button' aria-label='Voice control' tabindex='0'>
       <svg viewBox='0 0 120 60' width='46' height='24'>
           <polyline points='0,40 15,30 30,35 45,20 60,28 75,18 90,32 105,26 120,38' fill='none' stroke='white' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/>
       </svg>
    </div>
    <script>
       const doc = window.parent.document;
       const fab = doc.getElementById('fab-mic');
       function triggerHidden(){
           const buttons = Array.from(doc.querySelectorAll('button'));
           const hidden = buttons.find(b => b.innerText.trim() === 'Voice Record Toggle');
           if(hidden){ hidden.click(); }
       }
       if(fab){
           fab.addEventListener('click', triggerHidden);
           fab.addEventListener('keypress', (e)=>{ if(e.key==='Enter' || e.key===' '){ e.preventDefault(); triggerHidden(); }});
       }
    </script>
    """.replace('{STATE}', mic_state_class)
    st.markdown(mic_html, unsafe_allow_html=True)

with st.expander("Session History"):
    for i,h in enumerate(st.session_state.get('history', [])[-30:]):
        st.markdown(f"**{i+1}. {h['type']} :: {h.get('section','')}**")
        st.markdown(f"<div style='white-space:pre-line;font-size:.8rem;'>{clean_markdown(h['text'])}</div>", unsafe_allow_html=True)
        if h.get('audio_bytes'):
            st.audio(h['audio_bytes'], format='audio/wav')

st.markdown("<div style='margin-top:3rem;text-align:center;opacity:.55;font-size:.7rem;'>VocaResume – Modern Interface</div>", unsafe_allow_html=True)
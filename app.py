"""Main Streamlit application for VocaResume (refactored).

This module provides the streamlined resume analysis + optional TTS narration UI.
Speech-to-text recording has been fully removed. Voice toggle now only controls
automatic TTS playback of generated summaries.
"""

from __future__ import annotations

# Standard libs
import os, re, hashlib, base64

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
from tasks.task_factory import create_tasks
from tasks.vector_router import init_router
from agents.agent_factory import create_agents
from utils.voice_utils import text_to_speech, voice_enabled, voice_stack_report
from utils.file_utils import input_pdf_setup
from utils.text_utils import clean_markdown


def _init_session_state():
    st.session_state.setdefault('app_view', 'landing')
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('responses', {})
    st.session_state.setdefault('voice_played', {})


def _resolve_logo():
    for name in ["vocaresume_logo.png","logo.png","VocaResume.png"]:
        p = os.path.join("static", name)
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str | None:
    path = _resolve_logo()
    if not path: return None
    try:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        ext = 'png' if path.lower().endswith('.png') else 'jpg'
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return None


def _render_landing_page():
    logo = _logo_data_uri()
    st.markdown(
        """
<style>
/* --- Landing Page Modern Styles --- */
body, .stApp {background:radial-gradient(circle at 20% 20%, #f2f9ff 0%, #ffffff 55%, #f3fbfd 100%) !important;}
.landing-shell{max-width:1180px;margin:0 auto;padding:3.2rem 2.2rem 4rem;}
.hero{display:flex;flex-direction:column;align-items:center;text-align:center;gap:1.2rem;position:relative;}
.hero h1{font-size:3.2rem;background:linear-gradient(90deg,#0b7285,#1098ad);-webkit-background-clip:text;color:transparent;margin:0;font-weight:700;letter-spacing:.5px;}
.hero-tag{font-size:1.15rem;max-width:780px;line-height:1.45;color:#335360;margin:0 auto;font-weight:400;}
.hero-badges{display:flex;gap:.75rem;flex-wrap:wrap;justify-content:center;margin-top:.5rem;}
.badge{background:#e3f8fb;color:#0b7285;padding:.45rem .85rem;font-size:.7rem;font-weight:600;letter-spacing:.75px;border-radius:40px;text-transform:uppercase;border:1px solid #cff4f9;}
.cta-primary{background:#0b7285;color:#fff;border:none;padding:.95rem 1.6rem;font-size:1rem;font-weight:600;border-radius:14px;cursor:pointer;box-shadow:0 4px 18px -4px rgba(11,114,133,.45);transition:.25s ease;display:inline-flex;align-items:center;gap:.55rem;}
.cta-primary:hover{background:#095d6b;transform:translateY(-2px);box-shadow:0 8px 24px -6px rgba(11,114,133,.55);} 
.logo-landing{width:86px;height:86px;object-fit:contain;filter:drop-shadow(0 8px 22px rgba(11,114,133,.35));animation:floatLogo 6s ease-in-out infinite;} 
@keyframes floatLogo{0%,100%{transform:translateY(0);}50%{transform:translateY(-10px);}}

/* Feature Cards */
.features-grid{margin-top:3.5rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.35rem;}
.f-card{position:relative;background:linear-gradient(145deg,#ffffff, #f0fbfd);border:1px solid #d8eef2;border-radius:20px;padding:1.15rem 1.15rem 1.35rem;box-shadow:0 6px 18px -4px rgba(15,109,126,.08);overflow:hidden;min-height:180px;display:flex;flex-direction:column;gap:.55rem;}
.f-card:before{content:"";position:absolute;inset:0;background:radial-gradient(circle at 85% 10%,rgba(16,152,173,.18),transparent 60%);opacity:.9;pointer-events:none;}
.f-card h3{margin:0;font-size:.95rem;letter-spacing:.5px;font-weight:600;color:#0b7285;display:flex;align-items:center;gap:.4rem;}
.f-card p{margin:0;font-size:.78rem;line-height:1.35;color:#3d5661;font-weight:400;}
.f-icon{width:34px;height:34px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:.9rem;background:#0b7285;color:#fff;box-shadow:0 4px 10px -3px rgba(11,114,133,.5);}


/* Steps */
.steps{margin-top:4rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;}
.step{background:#ffffff;border:1px solid #d9eef1;border-radius:18px;padding:1rem .95rem;display:flex;flex-direction:column;gap:.4rem;position:relative;box-shadow:0 4px 14px -4px rgba(15,109,126,.08);} 
.step-num{width:34px;height:34px;border-radius:12px;background:#0b7285;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:600;font-size:.8rem;box-shadow:0 4px 12px -4px rgba(11,114,133,.5);} 
.step h4{margin:.2rem 0 0;font-size:.8rem;letter-spacing:.6px;text-transform:uppercase;color:#0b7285;} 
.step p{margin:0;font-size:.7rem;line-height:1.3;color:#35505a;}

.footer-landing{margin-top:4.5rem;text-align:center;font-size:.7rem;letter-spacing:.5px;color:#51737d;opacity:.75;}
@media (max-width:760px){.hero h1{font-size:2.35rem;}.features-grid{grid-template-columns:1fr 1fr;}.c-quote{font-size:.95rem;}}
</style>
""",
        unsafe_allow_html=True
    )

    st.markdown("<div class='landing-shell'>", unsafe_allow_html=True)
    # Hero Section
    st.markdown(
        f"""
        <div class='hero'>
            {f'<img src="{logo}" class="logo-landing" alt="Logo" />' if logo else ''}
            <div class='hero-badges'>
                <span class='badge'>AI DRIVEN</span>
                <span class='badge'>VOICE READY</span>
                <span class='badge'>VECTOR ROUTED</span>
            </div>
            <h1>VocaResume</h1>
            <p class='hero-tag'>Accelerate your job search with intelligent resume analysis, interview prep, optimization tips and instant spoken summaries – all in one streamlined workspace.</p>
            <button class='cta-primary' onclick="window.location.href='?view=main'">Launch App →</button>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature Cards
    st.markdown("<div class='features-grid'>", unsafe_allow_html=True)
    features = [
        ("Semantic Task Routing","Understands your intent (analysis, interview, suggestions, fit scoring) with a resilient vector engine."),
        ("Doc Intelligence","High-fidelity text extraction for sharper insights – no more blank PDFs."),
        ("Adaptive Voice","Conversational script planning + natural TTS for instant audio briefings."),
        ("Optimization Insights","Actionable rewrite suggestions & improvement deltas to level up your resume."),
        ("Interview Generation","Adaptive technical & behavioral questions generated from your experience."),
        ("Fit Scoring","Multi-factor role–candidate alignment scoring with rationale.")
    ]
    for title, desc in features:
        st.markdown(f"<div class='f-card'><div class='f-icon'>★</div><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # Steps Section
    st.markdown("<div class='steps'>", unsafe_allow_html=True)
    steps = [
        ("Upload","Drop in your resume PDF."),
        ("Describe","Paste role or job description."),
        ("Ask","Query: analyze, optimize, interview, fit."),
        ("Listen","Get spoken summary instantly."),
    ]
    for i,(t,d) in enumerate(steps, start=1):
        st.markdown(f"<div class='step'><div class='step-num'>{i}</div><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer-landing'>VocaResume · Built for modern career navigation · © 2025</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Streamlit button fallback (keyboard access) hidden visually but present for a11y
    if st.button("Launch App", key="enter_app_btn", help="Open main interface"):
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

# Sticky Glass Header (logo) for main app
logo_data_uri = _logo_data_uri()
if not logo_data_uri:
    st.warning("Logo image missing (checked vocaresume_logo.png, logo.png, VocaResume.png)")

st.markdown("""
<style>
.main-header{margin:1rem auto 1.25rem;max-width:980px;text-align:center;position:relative;padding-top:.4rem;}
.main-header .logo{width:70px;height:70px;object-fit:contain;border-radius:22px;background:#e0f7fb;padding:6px;box-shadow:0 6px 20px -6px rgba(11,114,133,.4);} 
.main-header h1{margin:.8rem 0 0;font-size:2.1rem;font-weight:700;letter-spacing:.5px;background:linear-gradient(90deg,#0b7285,#1098ad);-webkit-background-clip:text;color:transparent;}
.main-header .tag{margin:.2rem 0 0;font-size:.7rem;letter-spacing:1.5px;font-weight:600;color:#147f91;opacity:.85;}
.control-row{max-width:1100px;margin:0 auto 1rem;display:grid;grid-template-columns:280px 120px 1fr 90px;gap:1rem;align-items:end;}
.ctrl-label{font-size:.6rem;letter-spacing:1.1px;text-transform:uppercase;font-weight:600;color:#0b7285;opacity:.7;margin-bottom:.25rem;}
.token-slider .stSlider{padding-top:0 !important;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='main-header'>
  <img src='{logo_data_uri or ''}' class='logo' alt='logo'/>
  <h1>VocaResume</h1>
  <div class='tag'>CAREER COPILOT</div>
</div>
""", unsafe_allow_html=True)

# --- Control Bar (restructured) ---
st.markdown(
    """
    <style>
    .control-bar-wrapper{max-width:1080px;margin:0 auto 1.4rem;padding:.85rem 1rem;border:1px solid #d9ecef;background:linear-gradient(145deg,#ffffff,#f3fbfd);border-radius:20px;box-shadow:0 4px 14px -6px rgba(11,114,133,.15);}
    .control-grid{display:grid;grid-template-columns: minmax(240px,310px) 110px minmax(320px,1fr) 90px;gap:1rem;align-items:end;}
    @media (max-width:980px){.control-grid{grid-template-columns:1fr 1fr;}}
    .ctrl-block{display:flex;flex-direction:column;gap:.35rem;}
    .ctrl-title{font-size:.58rem;letter-spacing:1px;font-weight:600;text-transform:uppercase;color:#0b7285;opacity:.78;}
    .voice-toggle-holder{display:flex;align-items:center;justify-content:flex-start;gap:.5rem;padding:.35rem .55rem;border:1px solid #cfe7ec;background:#eef9fb;border-radius:14px;}
    .voice-toggle-holder span{font-size:.6rem;letter-spacing:.8px;font-weight:600;color:#0b7285;}
    /* Hide internal toggle label */
    div[data-testid="stVerticalBlock"] .voice-toggle-holder label{display:none !important;}
    .token-slider-holder{padding:.4rem .65rem .2rem;border:1px solid #d3e9ed;background:#f6fcfd;border-radius:14px;}
    .token-value{font-size:.55rem;letter-spacing:.5px;color:#0b7285;font-weight:600;display:inline-block;margin-left:.4rem;}
    .cache-btn-holder{text-align:right;}
    .cache-btn-holder button{width:100%;background:#ffecec;color:#c92a2a;border:1px solid #ffc9c9;}
    .cache-btn-holder button:hover{background:#ffc9c9;color:#801b1b;}
    .section-divider{max-width:1080px;margin:0.2rem auto 0.8rem;border:none;height:1px;background:linear-gradient(90deg,rgba(11,114,133,.15),rgba(11,114,133,.4),rgba(11,114,133,.1));}
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown("<div class='control-bar-wrapper'><div class='control-grid'>", unsafe_allow_html=True)
    # Model selector
    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        st.markdown("<div class='ctrl-block'><div class='ctrl-title'>Model</div>", unsafe_allow_html=True)
        model_choice = st.selectbox("Model", list(settings.AVAILABLE_MODELS.keys()), index=0, key="model_select", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='ctrl-block'><div class='ctrl-title'>Voice</div>", unsafe_allow_html=True)
        # Voice toggle w/out textual label; accessible label in help
        voice_mode = st.toggle("", value=False, key="voice_toggle", help="Toggle spoken responses on/off", label_visibility="collapsed")
        # Surround toggle with styled holder (inject minimal script to move element?) simpler: just caption indicator
        st.caption("On" if voice_mode else "Off")
        st.markdown("</div>", unsafe_allow_html=True)
    with colC:
        st.markdown("<div class='ctrl-block'><div class='ctrl-title'>Max Tokens <span class='token-value'>"+str(st.session_state.get('max_tokens_slider',4096))+"</span></div>", unsafe_allow_html=True)
        max_tokens = st.slider("Tokens", 512, 8192, 4096, 256, key="max_tokens_slider", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    with colD:
        st.markdown("<div class='ctrl-block'><div class='ctrl-title'>Cache</div>", unsafe_allow_html=True)
        if st.button("Clear", key="clear_cache_btn"):
            st.session_state.get('responses', {}).clear()
            st.toast("Cache Cleared")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)

# Expose model configuration selected in control bar (robust to partial reruns)
model_info = settings.AVAILABLE_MODELS.get(model_choice, {}) if 'model_choice' in locals() else {}
if not model_info or not model_info.get('provider') or not model_info.get('model'):
    st.error("Model config error: provider or model missing in AVAILABLE_MODELS entry.")

st.markdown("""
<style>
/* Modern minimalist layout */
.layout-grid {display:grid;grid-template-columns:420px 1fr;gap:1.75rem;align-items:start;margin-top:1.25rem;}
@media (max-width:1150px){.layout-grid{grid-template-columns:1fr;}}
.panel {background:rgba(255,255,255,0.55);backdrop-filter:blur(14px);border:1px solid rgba(0,0,0,.08);border-radius:18px;padding:1.15rem 1.25rem;box-shadow:0 4px 14px -4px rgba(0,0,0,.07);} 
.panel h3 {margin:0 0 .75rem;font-size:1.05rem;font-weight:600;letter-spacing:.5px;display:flex;align-items:center;gap:.5rem;}
.panel small{opacity:.75;}
.btn-primary {background:#0b7285;color:#fff;border:none;padding:.65rem 1.1rem;border-radius:10px;font-weight:500;font-size:.9rem;cursor:pointer;display:inline-flex;align-items:center;gap:.5rem;}
.status-tag {display:inline-block;padding:.25rem .55rem;border-radius:999px;font-size:.65rem;letter-spacing:.5px;text-transform:uppercase;background:#0b7285;color:#fff;margin-left:.5rem;}
.output-block {background:#0d1117;color:#d0d4da;font-family:var(--font-mono, ui-monospace, Menlo, monospace);padding:1rem;border-radius:14px;font-size:.8rem;line-height:1.45;position:relative;}
.output-block h4{margin:.25rem 0 .75rem;font-size:.8rem;text-transform:uppercase;letter-spacing:.75px;font-weight:600;color:#91d1e0;}
.pill {display:inline-flex;align-items:center;font-size:.65rem;padding:.2rem .55rem;border-radius:16px;letter-spacing:.5px;background:#eef8f9;color:#0b7285;margin-right:.4rem;font-weight:500;}
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
    if 'vector_router' not in st.session_state:
        try:
            st.session_state['vector_router'] = init_router(persist_dir=".chroma_store")
        except Exception as e:
            st.warning(f"Vector router unavailable: {e}")
    if st.button("Generate", type="primary", use_container_width=True, disabled=not (input_text and uploaded_file and user_query)):
        st.session_state['last_run'] = {'input_text': input_text, 'user_query': user_query}
        # Use vector router to decide task
        vr = st.session_state.get('vector_router')
        if vr:
            try:
                # ingest sources (idempotent upserts)
                st.session_state.setdefault('ingested_once', False)
                if not st.session_state['ingested_once'] and uploaded_file:
                    # resume ingested later after parsing for text; store flag
                    st.session_state['pending_ingest_resume'] = True
                vr.ingest_job_description(input_text)
                route_res = vr.route(user_query)
                st.session_state['detected_task'] = {'index': route_res.task_index, 'intent': route_res.label, 'score': route_res.score, 'alt': route_res.alt}
            except Exception as e:
                st.error(f"Routing failed: {e}")
        else:
            st.session_state['detected_task'] = {'index': 0, 'intent': 'analysis', 'score': 0.0, 'alt': []}
    st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT PANEL: Outputs
    st.markdown("<div class='panel' id='right-panel'>", unsafe_allow_html=True)
    # Determine routing backend badge
    backend_badge = ""
    vr = st.session_state.get('vector_router')
    stats_html = ''
    if vr:
        backend = getattr(vr, '_routing_backend', 'chroma')
        badge_color = '#0b7285' if backend == 'chroma' else '#ff8800'
        backend_badge = f"<span class='status-tag' style='background:{badge_color};'>{backend.upper()}</span>"
        try:
            stats = vr.stats()  # type: ignore[attr-defined]
            if stats:
                # show top 2 counts compact
                items = sorted(stats.items(), key=lambda x: x[1], reverse=True)
                top = ', '.join([f"{k}:{v}" for k,v in items[:2]])
                stats_html = f"<span style='margin-left:.5rem;font-size:.55rem;letter-spacing:.5px;opacity:.55;'>[{top}]</span>"
        except Exception:
            pass
    st.markdown(f"<h3>Output <span class='status-tag'>RESULT</span>{backend_badge}{stats_html}</h3>", unsafe_allow_html=True)
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
                # ingest resume text into vector router once
                if pdf_content and st.session_state.get('pending_ingest_resume'):
                    vr = st.session_state.get('vector_router')
                    if vr:
                        try:
                            resume_text = pdf_content[0]['data']
                            vr.ingest_resume(resume_text)
                            st.session_state['ingested_once'] = True
                            st.session_state['pending_ingest_resume'] = False
                        except Exception as e:
                            st.warning(f"Resume ingest failed: {e}")
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
                    # Graceful fallback: derive a concise spoken script from cleaned text
                    fallback_script = cleaned.split('\n')[:8]
                    fallback_script = ' '.join([re.sub(r'\s+',' ', ln).strip() for ln in fallback_script])[:800]
                    st.warning("Script planner degraded – using fallback summarization for voice.")
                    st.session_state['last_pipeline'] = {
                        'task': intent,
                        'raw': raw_md,
                        'cleaned': cleaned,
                        'plan': '',
                        'script': fallback_script or cleaned[:600]
                    }
                else:
                    st.session_state['last_pipeline'] = {
                        'task': intent,
                        'raw': raw_md,
                        'cleaned': cleaned,
                        'plan': script_plan.get('plan',''),  # retained for debug only
                        'script': script_plan.get('script','')
                    }
    pipe = st.session_state.get('last_pipeline')
    if pipe:
        vr = st.session_state.get('vector_router')
        backend = getattr(vr, '_routing_backend', 'chroma') if vr else 'keyword'
        # Show routing score and alt candidates if available
        det = st.session_state.get('detected_task', {})
        score = det.get('score')
        alt = det.get('alt', []) or []
        alt_str = ' '.join([f"{a['label']}({a['score']:.2f})" for a in alt[:3]]) if alt else ''
        score_html = f"<span style='margin-left:.4rem;opacity:.65;'>score {score:.2f}</span>" if score is not None else ''
        alt_html = f"<div style='font-size:.6rem;opacity:.6;margin-top:.2rem;'>Alt: {alt_str}</div>" if alt_str else ''
        st.markdown(f"<div class='pill'>Task: {pipe['task']} · Route: {backend}{score_html}</div>{alt_html}", unsafe_allow_html=True)
        # Only show final analysis result (raw LLM output)
        st.markdown(f"<div class='output-block'><h4>RESULT</h4>{clean_markdown(pipe['raw'])}</div>", unsafe_allow_html=True)
        if voice_mode and pipe.get('script'):
            audio_bytes = text_to_speech(pipe['script'][:1800])
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mpeg')
                st.download_button("Download Audio", data=audio_bytes, file_name="spoken.mp3", mime="audio/mpeg", use_container_width=True)
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
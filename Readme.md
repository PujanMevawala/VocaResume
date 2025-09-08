# VocaResume

Modern, voice-enabled career copilot for resume intelligence: analyze strengths vs. a job description, generate optimization suggestions, produce tailored interview questions, and calculate a job fit score—now in a lean 5‑folder structure.

> Rebranded to **VocaResume** (formerly SmartFitAI / TalentAlign AI).

## ✨ Features

| Area               | Highlights                                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| Resume Analysis    | Structured strengths & weaknesses grounded in JD context                                                    |
| Suggestions        | Actionable ATS + clarity improvements                                                                       |
| Interview Prep     | 5 technical, resume-grounded Q&A items                                                                      |
| Job Fit Score      | Numeric score + qualitative rationale + gauge viz                                                           |
| Multi‑Provider LLM | Gemini (Pro / Flash), Groq LLaMA variants, Perplexity Sonar                                                 |
| Voice Mode         | Record (browser) → STT (Vosk or Whisper API) → intent route → TTS (EdgeTTS → fallback gTTS/pyttsx3/browser) |
| Robust UI          | Landing page + single-page workflow, accessible components                                                  |
| Caching            | Per (section, model) response memory for speed                                                              |

## 🗂 Current Structure (Post-Cleanup)

```
.
├── app.py                  # Streamlit app (landing + main tool)
├── services/               # Core service layer
│   ├── settings.py         # Model registry & API keys
│   ├── model_service.py    # Provider dispatch (Google/Groq/Perplexity)
│   ├── agent_factory.py    # CrewAI agent initialization
│   └── task_factory.py     # Task descriptions + voice intent routing
├── utils/                  # Supporting utilities
│   ├── file_utils.py       # PDF ingestion helpers
│   ├── style_utils.py      # CSS loader
│   ├── text_utils.py       # Markdown cleanup for TTS/display
│   ├── voice_utils.py      # Record + STT + TTS orchestration
│   └── stt_providers.py    # Pluggable STT provider selection
├── static/                 # CSS + images
├── models/                 # (Optional) local models (e.g., Vosk)
├── tests/                  # Pytest smoke & settings tests
├── requirements.txt
├── render.yaml             # Render deployment config
├── config.toml             # Streamlit theme/server config
└── README.md
```

Removed unused legacy modules: `llm_utils.py`, `image_utils.py`, `format_utils.py`, duplicate agent/task/ui legacy folders, old `config/settings.py` (shim retained for backward import).

## 🔑 Environment Variables

```
GROQ_API_KEY=...
GOOGLE_API_KEY=...
PPLX_API_KEY=...             # Optional (Perplexity Sonar)
VOSK_MODEL_PATH=./models/vosk
WHISPER_API_ENABLED=1        # (optional) use remote Whisper if implemented
WHISPER_MODEL=whisper-1
DISABLE_OFFLINE_TTS=1        # Skip pyttsx3; use gTTS/browser
VOSK_SKIP_DL=1               # Faster cold start (no auto download)
VOICE_DEBUG=0                # Set 1 for diagnostics pane
```

Voice mode gracefully degrades: no Vosk → STT disabled, TTS still available.

## � Quick Start

```bash
git clone <repo-url>
cd SmartFit
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env  # (if you have a template) then edit with keys
streamlit run app.py
```

### Optional: Add Vosk Model

```bash
mkdir -p models/vosk
curl -L -o vosk-small.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-small.zip -d models/vosk && rm vosk-small.zip
```

## 🧪 Testing

```bash
pytest -q
```

Tests include: Streamlit load smoke test + settings/env structure validation.

## 🔊 Voice / STT / TTS Flow (Updated)

1. Record via `streamlit-audiorecorder` (browser mic)
2. STT provider selection (`stt_providers.py`): Whisper API (planned) → local Vosk (if installed)
3. Intent routing (voice query → task)
4. LLM response generation (selected provider/model)
5. Markdown is split into two forms:
   - `display_md`: rich markdown for on‑screen display
   - `tts_text`: sanitized plain text produced by `normalize_for_tts()` (no `#`, `*`, code fences)
6. TTS cascade: EdgeTTS (neural voices, MP3 @ 22.05 kHz) → gTTS → pyttsx3 (legacy) → browser speech synthesis
7. Audio file stored atomically and offered for playback + download; session history retains recent entries.

All steps are fail-soft—failures log + degrade without crashing the UI.

### Markdown Sanitization

Implemented in `utils/text_utils.py::normalize_for_tts` using `mistune` + `BeautifulSoup`:

| Markdown Input             | Sanitized Output (excerpt) |
| -------------------------- | -------------------------- |
| `## **Resume Analysis**`   | `Resume Analysis:`         |
| Bullet list (`- item one`) | `• item one`               |
| Code block ( > 80 chars )  | `(code block omitted)`     |

The sanitizer collapses whitespace, strips formatting tokens, summarizes or removes verbose code, and truncates long text near a sentence boundary.

### Before / After Example

Input Markdown:

````
## **Resume Analysis** ##
- Strengths: *Python*, **Data Pipelines**
- Weaknesses: Limited cloud exp.
```python\nprint('debug')\n```

Overall fit: **High**
````

EdgeTTS Spoken Text:

```
Resume Analysis: • Strengths: Python, Data Pipelines • Weaknesses: Limited cloud exp. (code block omitted) Overall fit: High
```

No raw markdown tokens remain, eliminating prior artifacts in MP3 downloads.

## 🧱 Deployment (Render)

Native (light) or Docker (full voice):

| Mode   | Use When                            | Pros         | Cons                         |
| ------ | ----------------------------------- | ------------ | ---------------------------- |
| Native | Basic LLM + optional browser speech | Fast build   | No offline STT / offline TTS |
| Docker | Need offline STT/TTS                | Full control | Longer build                 |

Ensure these in Render (native):

```
Build: pip install -r requirements.txt
Start: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Set env vars (keys + DISABLE_OFFLINE_TTS=1, VOSK_SKIP_DL=1).

For Docker: supply a Dockerfile adding `ffmpeg`, `espeak`, `poppler-utils`, copy model if desired; unset DISABLE_OFFLINE_TTS.

## 🧹 Caching Strategy

Responses keyed by (section_id, model_choice). Extension idea: hash of JD + resume bytes for automatic invalidation when inputs change.

## 🔐 Security

- Do not commit `.env`
- Rotate API keys
- Restrict available models in `services/settings.py` if minimizing cost

## 📝 Git Flow

```bash
git checkout -b feat/new-voice-provider
# changes
git add .
git commit -m "feat: add faster-whisper provider"
git push origin feat/new-voice-provider
```

Use conventional commits (`feat:`, `fix:`, `chore:`, `docs:` ...). Tag releases when stable:

```bash
git tag -a v1.0.0 -m "Initial cleaned structure"
git push origin v1.0.0
```

## ❓ Troubleshooting

| Symptom          | Likely Cause                      | Action                                   |
| ---------------- | --------------------------------- | ---------------------------------------- |
| No transcription | Missing Vosk model / STT disabled | Install model or enable Whisper API      |
| 0:00 audio       | pyttsx3/EdgeTTS failure           | Check logs (VOICE_DEBUG=1) fallback used |
| PDF error        | Poppler missing                   | Use Docker or switch to pure text input  |
| Slow response    | Large token limit / heavy model   | Lower Max Tokens slider                  |

## © 2025 VocaResume

Focused, modular, voice-capable career intelligence.

# VocaResume (formerly TalentAlign AI / SmartFitAI)

Modern, voice‑enabled career copilot: analyze resumes, optimize content, generate interview questions, and compute job fit—all in a single streamlined Streamlit app.

> Project has been rebranded to **VocaResume**. Previous names: *TalentAlign AI*, *SmartFitAI*. Update any existing Render service & Git remote to reflect the new name (see Git workflow section).

## ✨ Core Features

| Capability | Description |
|------------|-------------|
| Resume Analysis | Structured strengths / weaknesses vs. target JD |
| Suggestions | Actionable optimization guide (ATS + clarity) |
| Interview Prep | 5 resume‑grounded technical Q&A pairs |
| Job Fit Score | Quantitative score + qualitative rationale |
| Multi‑Model Routing | Gemini (Pro/Flash), Groq LLaMA, Perplexity Sonar |
| Voice Mode | Offline-first record → STT (Vosk) → intent routing → optional TTS (pyttsx3) |
| Accessible UI | Keyboard focus, ARIA roles, reduced‑motion support |

## 🗂 Project Structure (Trimmed & Cleaned)

```
SmartFit/
├── app.py                   # Main Streamlit application (landing + single-page UX)
├── config/
│   └── settings.py          # Model registry & API key loading
├── services/
│   └── model_service.py     # Provider dispatch (Google/Groq/Perplexity)
├── agents/
│   └── agent_factory.py     # CrewAI agent construction
├── tasks/
│   └── task_factory.py      # Task definitions + voice intent routing
├── utils/
│   ├── file_utils.py        # PDF → first page image (Gemini multimodal input)
│   ├── style_utils.py       # CSS loader with graceful fallback
│   ├── text_utils.py        # Markdown cleanup for display
│   └── voice_utils.py       # Record, STT (Vosk), TTS (pyttsx3)
├── static/
│   ├── style.css            # Main theme & component styles (voice recorder)
│   └── * assets *.png/svg   # Inline‑encoded landing icons
├── models/vosk/             # (Optional) Local Vosk model for offline STT
├── tests/                   # Basic smoke & config tests
├── requirements.txt
├── render.yaml              # Render deployment config
├── config.toml              # Streamlit server/theme overrides
└── README.md
```

Removed legacy / unused modules: `ui_helpers.py`, `format_utils.py`, `image_utils.py`, `llm_utils.py`, duplicate logging config, unused benchmarking util.

## 🔑 Environment Variables (.env)

```
GROQ_API_KEY=...
GOOGLE_API_KEY=...
PPLX_API_KEY=...        # Perplexity (optional if not using Sonar models)
VOSK_MODEL_PATH=./models/vosk   # (optional) if custom location
```

Voice mode still works (TTS) without Vosk; STT just disabled if model absent.

## 🛠 Local Development

```bash
git clone <your-repo-url>
cd SmartFit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env  # (create and fill if example provided)
streamlit run app.py
```

### Optional: Download a Small Vosk Model
```bash
mkdir -p models/vosk
curl -L -o vosk-small.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-small.zip -d models/vosk && rm vosk-small.zip
```

## 🚀 Deploying on Render

1. Push latest code to GitHub (see Git workflow below).
2. Create a new Web Service → point to repo root.
3. Build Command: `pip install -r requirements.txt`
4. Start Command:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add Environment Variables (same as local). Set `PORT=8501` (Render injects it; still safe).
6. (Optional) Add a persistent disk if you intend to store downloaded Vosk model at runtime.

`render.yaml` may further simplify provisioning if you enable auto deploy on push.

## 🔊 Voice Mode Architecture

1. Browser audio capture via `streamlit-audiorecorder` → WAV bytes
2. STT: Vosk (offline) converts audio → text (if model present)
3. Intent routing (`get_task_from_query`) chooses task (analysis / suggestions / interview / job fit)
4. Response optionally synthesized with pyttsx3 (offline TTS)

All stages fail gracefully without crashing UI.

## 🧪 Tests

Run basic tests:
```bash
pytest -q
```

Current tests: app smoke load, settings structure & env variable loading.

## 🧹 Maintenance & Caching

Responses cached per (section_id, model_choice). Future enhancement: include content hash of JD + resume image for stronger invalidation.

## 🔐 Security & Secrets

- Never commit `.env`
- Rotate API keys periodically
- Limit model usage to necessary providers (adjust `AVAILABLE_MODELS`)

## 📝 Git Workflow (Recommended)

```bash
git checkout -b chore/cleanup-voice-ui
# make changes
git add .
git commit -m "chore: streamline voice recorder + remove unused modules"
git push origin chore/cleanup-voice-ui
```
Open a PR → squash & merge → Render auto-deploy (if configured). For direct main pushes use conventional commit prefixes:`feat:, fix:, chore:, refactor:, docs:, perf:`.

Tag a release (optional):
```bash
git tag -a v1.0.0 -m "Initial streamlined UI & voice integration"
git push origin v1.0.0
```

## ❓ Troubleshooting

| Issue | Fix |
|-------|-----|
| No STT transcription | Ensure Vosk model folder exists & matches sample rate |
| TTS silent | `pyttsx3` engine may fail on headless Linux; install `espeak`/`ffmpeg` |
| PDF processing error | Ensure `poppler` available (Render base image sometimes lacks). Consider switching to pure text ingestion in future |
| API timeout | Reduce `max_tokens` slider or switch model |

## © 2025 TalentAlign AI

Crafted for efficient, multimodal career intelligence.

## Local Development Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd SmartFit
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   PPLX_API_KEY=your_perplexity_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Deployment to Render

### Prerequisites

- GitHub account with your code repository
- Render account
- API keys for Groq and Google Generative AI

### Deployment Steps

1. **Push your code to GitHub**

   ```bash
   git add .
   git commit -m "Initial deployment setup"
   git push origin main
   ```

2. **Connect to Render**

   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" and select "Web Service"

3. **Configure the service**

   - **Name**: `smartfit` (or your preferred name)
   - **Repository**: Select your GitHub repository
   - **Branch**: `main`
   - **Root Directory**: Leave empty (if app is in root)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Set Environment Variables**
   In Render dashboard, go to your service → Environment:

   - `GROQ_API_KEY`: Your Groq API key
   - `GOOGLE_API_KEY`: Your Google Generative AI API key
   - `PPLX_API_KEY`: Your Perplexity API key
   - `PORT`: `8501`

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Your app will be available at the provided URL

## API Keys Setup

### Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up/Login
3. Navigate to API Keys section
4. Create a new API key
5. Copy and use in your environment variables

### Google Generative AI API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Navigate to API Keys
4. Create a new API key
5. Copy and use in your environment variables

### Perplexity API Key

1. Go to [`https://www.perplexity.ai/settings/api`](https://www.perplexity.ai/settings/api)
2. Create an API key
3. Use as `PPLX_API_KEY`

## File Structure

```
SmartFit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment configuration
├── config.toml           # Streamlit configuration
├── smartfit_logo.jpg     # Application logo
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Security Notes

- Never commit `.env` files to version control
- API keys are stored as environment variables in Render
- The `.gitignore` file excludes sensitive files from Git

## Troubleshooting

### Common Issues

1. **App not loading**: Check if all environment variables are set in Render
2. **PDF processing errors**: Ensure `poppler-utils` is properly installed
3. **API errors**: Verify your API keys are valid and have sufficient credits

### Local Development Issues

1. **Port conflicts**: Change port in `config.toml` if 8501 is in use
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Environment variables**: Ensure `.env` file exists with proper API keys

## Support

For issues and questions:

- Check the troubleshooting section above
- Review Render deployment logs
- Ensure all dependencies are properly installed

---

© 2025 SmartFitAI 🤖 - Matching You Smartly to Your Dream Job!

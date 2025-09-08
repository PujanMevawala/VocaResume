# Dockerfile for full voice + TTS (EdgeTTS primary) deployment on Render / container platform.
# Provides ffmpeg (audio normalization) + espeak (legacy pyttsx3 fallback) + poppler-utils (pdf2image)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: ffmpeg (audio), espeak (pyttsx3 on Linux), poppler-utils (pdf2image) unzip curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ffmpeg espeak poppler-utils curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .

# Optional: pre-download small Vosk model to reduce cold start; comment out if using persistent disk
# RUN mkdir -p models/vosk \
#  && curl -L -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip \
#  && unzip /tmp/vosk.zip -d models/vosk \
#  && rm /tmp/vosk.zip \
#  && touch models/vosk/.installed

# Default environment for full stack
ENV DISABLE_OFFLINE_TTS=0 \
    VOSK_SKIP_DL=0 \
    VOICE_DEBUG=0

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

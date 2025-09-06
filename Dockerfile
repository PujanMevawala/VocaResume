FROM python:3.11-slim

# Install system dependencies (audio + PDF + build tools)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak \
    libespeak1 \
    libespeak-dev \
    portaudio19-dev \
    poppler-utils \
    build-essential \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip & install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

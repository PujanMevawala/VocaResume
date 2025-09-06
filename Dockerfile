FROM python:3.11-slim

# Install system dependencies (audio + build tools)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak \
    libespeak1 \
    portaudio19-dev \
    build-essential \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip & install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (for Streamlit)
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

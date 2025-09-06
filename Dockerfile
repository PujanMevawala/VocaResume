FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg espeak && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (change if your app uses a different port)
EXPOSE 8501

# Command to run your app (update as needed)
CMD ["python", "app.py"]
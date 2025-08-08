# services/voice/Dockerfile
# Updated to use the CUDA 12.9.0 runtime environment
#FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

# Install system dependencies required for audio processing and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Use a venv (keeps system Python clean)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Pip must be new enough to handle manylinux_2_28 wheels from PyTorch
RUN python -m pip install --upgrade pip setuptools wheel


WORKDIR /app

# Copy and install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the Whisper model during the build to speed up startup time
RUN python3 -c "import whisper; whisper.load_model('base.en')"

# Copy the application code
COPY app.py .

# Define the command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]

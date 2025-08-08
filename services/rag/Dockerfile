# services/rag/Dockerfile
#FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
# Install system dependencies, including Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use a venv (keeps system Python clean)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Pip must be new enough to handle manylinux_2_28 wheels from PyTorch
RUN python -m pip install --upgrade pip setuptools wheel

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Expose port 8001 for the service
EXPOSE 8001

# Command to run the application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get clean && apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    build-essential \
    git-lfs \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools for better compatibility
RUN pip install --upgrade pip setuptools wheel --no-cache-dir

# Install OpenCompass from source (more stable than PyPI)
# Use Python 3.10 compatible base image to avoid pyext compatibility issues
RUN git clone --depth 1 https://github.com/open-compass/opencompass.git /opencompass
WORKDIR /opencompass
RUN pip install -e . --no-cache-dir

# Install PEFT for adapter support
RUN pip install peft transformers accelerate --no-cache-dir

# Set working directory
WORKDIR /workspace

# Copy entrypoint script
COPY eval_entrypoint.py /app/eval_entrypoint.py


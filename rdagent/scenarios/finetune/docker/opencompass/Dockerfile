FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get clean && apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    build-essential \
    git-lfs \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCompass
RUN pip install opencompass

# Install PEFT for adapter support
RUN pip install peft transformers accelerate

# Set working directory
WORKDIR /workspace

# Copy entrypoint script
COPY eval_entrypoint.py /app/eval_entrypoint.py


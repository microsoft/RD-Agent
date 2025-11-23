FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get clean && apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    build-essential \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel --no-cache-dir

# Install OpenCompass with vLLM backend support
RUN git clone https://github.com/Jensen246/opencompass.git /opencompass
WORKDIR /opencompass

RUN pip install ".[vllm]" --no-cache-dir

# Install math evaluation dependencies for AIME/MATH benchmarks
RUN pip install math_verify latex2sympy2_extended --no-cache-dir

# Set working directory
WORKDIR /workspace

# Set environment variables for cache directories
ENV HF_HOME=/benchmarks/hf_cache
ENV HF_HUB_CACHE=/benchmarks/hf_cache/hub
ENV TRANSFORMERS_CACHE=/benchmarks/hf_cache/transformers
ENV HF_DATASETS_CACHE=/benchmarks/datasets
ENV COMPASS_DATA_CACHE=/benchmarks/opencompass_data

# Fix MKL threading layer compatibility issue with vLLM
ENV MKL_SERVICE_FORCE_INTEL=1
ENV MKL_THREADING_LAYER=GNU


FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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
RUN pip install opencompass vllm --no-cache-dir

# Install additional dependencies for LoRA evaluation and model support
RUN pip install peft transformers accelerate --no-cache-dir

# Set working directory
WORKDIR /workspace

# Copy evaluation scripts
COPY benchmark_config.py /app/benchmark_config.py
COPY opencompass_eval_entrypoint.sh /app/opencompass_eval_entrypoint.sh
RUN chmod +x /app/opencompass_eval_entrypoint.sh

# Set environment variables for cache directories
ENV HF_DATASETS_CACHE=/benchmarks
ENV COMPASS_DATA_CACHE=/benchmarks/opencompass_data
ENV HF_HOME=/benchmarks/hf_cache


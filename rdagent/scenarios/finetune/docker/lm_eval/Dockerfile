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

# Install lm-evaluation-harness with vLLM backend support
# Clone from GitHub to get latest tasks (including aime25)
RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git /lm-evaluation-harness
WORKDIR /lm-evaluation-harness
RUN pip install -e ".[vllm]" --no-cache-dir

# Install additional dependencies for LoRA evaluation
RUN pip install peft transformers accelerate --no-cache-dir

# Set working directory
WORKDIR /workspace

# Copy entrypoint script
COPY eval_entrypoint.sh /app/eval_entrypoint.sh
RUN chmod +x /app/eval_entrypoint.sh

# Set environment variable for HF datasets cache (mounted globally via extra_volumes)
ENV HF_DATASETS_CACHE=/benchmarks


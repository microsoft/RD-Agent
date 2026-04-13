# RD-Agent Unified Dockerfile
# Supports all scenarios: fin_factor, fin_quant, kaggle, data_science, RL, finetune
#
# Build commands:
#   CPU only:    docker build -f .devcontainer/Dockerfile -t rdagent:latest .
#   With GPU:    docker build -f .devcontainer/Dockerfile --build-arg GPU_SUPPORT=true -t rdagent:gpu .
#
# Proxy configuration (optional, for China users):
#   Without proxy: docker build -f .devcontainer/Dockerfile -t rdagent:latest .
#   With proxy:    docker build -f .devcontainer/Dockerfile -t rdagent:latest . \
#                    --build-arg HTTP_PROXY=http://your-proxy:port \
#                    --build-arg HTTPS_PROXY=http://your-proxy:port
#
# Run commands:
#   General:     docker run -it --rm -v $(pwd)/.env:/app/.env -v $(pwd)/workspace:/app/workspace rdagent:latest
#   fin_factor:  docker run -it --rm --env-file .env -v $(pwd)/workspace:/app/workspace rdagent:latest fin_factor
#   fin_quant:   docker run -it --rm --env-file .env -v $(pwd)/workspace:/app/workspace rdagent:latest fin_quant
#   kaggle:      docker run -it --rm --env-file .env -v $(pwd)/workspace:/app/workspace rdagent:latest kaggle
#   data_science:docker run -it --rm --env-file .env -v $(pwd)/workspace:/app/workspace rdagent:latest ds_loop

# ==============================================================================
# ARG definitions (must be before FROM)
# ==============================================================================

# Choose base image based on GPU support
ARG GPU_SUPPORT=false

# Proxy configuration (optional, passed at build time)
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG http_proxy=""
ARG https_proxy=""

# ==============================================================================
# Base image selection
# ==============================================================================
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS gpu-base
FROM python:3.10-slim AS cpu-base
FROM ${GPU_SUPPORT:+gpu-base}${GPU_SUPPORT:-cpu-base} AS base

FROM base

# Set proxy environment variables if provided
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ==============================================================================
# System dependencies
# ==============================================================================
RUN set -eux; \
    for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do \
        proxy_value=$(printenv "$proxy_var" || true); \
        case "$proxy_value" in \
            http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*) \
                unset "$proxy_var"; \
                ;; \
        esac; \
    done; \
    git config --global http.postBuffer 524288000 && \
    apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
        # Essential tools
        curl \
        vim \
        git \
        build-essential \
        coreutils \
        # Docker CLI (for container-in-container)
        docker.io \
        # Browser dependencies for web crawlers
        wget \
        gnupg \
        ca-certificates \
        # Font support for matplotlib
        fonts-liberation \
        libfontconfig1 \
        # HDF5 support (required for qlib data files)
        libhdf5-dev \
        libsnappy-dev \
        # Git LFS (for MLE-Bench and large datasets)
        git-lfs \
        # unzip for data processing
        unzip \
        # Clean up
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# ==============================================================================
# Google Chrome for web crawling
# ==============================================================================
RUN set -eux; \
    for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do \
        proxy_value=$(printenv "$proxy_var" || true); \
        case "$proxy_value" in \
            http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*) \
                unset "$proxy_var"; \
                ;; \
        esac; \
    done; \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver matching Chrome version
RUN CHROME_VERSION=$(google-chrome --version | awk '{print $2}' | cut -d '.' -f 1) \
    && CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION}") \
    && wget -q "https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip" \
    && unzip chromedriver_linux64.zip -d /usr/local/bin/ \
    && rm chromedriver_linux64.zip \
    && chmod +x /usr/local/bin/chromedriver

# ==============================================================================
# Python dependencies - Base (from requirements.txt)
# ==============================================================================
WORKDIR /app
COPY requirements.txt .

RUN set -eux; \
    for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do \
        proxy_value=$(printenv "$proxy_var" || true); \
        case "$proxy_value" in \
            http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*) \
                unset "$proxy_var"; \
                ;; \
        esac; \
    done; \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Python dependencies - fin_factor/fin_quant (Qlib scenario)
# ==============================================================================
RUN pip install --no-cache-dir \
    pyqlib \
    catboost \
    xgboost \
    tables

# Clone qlib repository for fin_quant scenario (needs specific commit)
RUN set -eux; \
    for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do \
        proxy_value=$(printenv "$proxy_var" || true); \
        case "$proxy_value" in \
            http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*) \
                unset "$proxy_var"; \
                ;; \
        esac; \
    done; \
    (git clone --depth 1 https://github.com/microsoft/qlib.git /opt/qlib_repo && \
     cd /opt/qlib_repo && \
     git fetch && \
     git reset 2fb9380b342556ddb50a4b24e4fe8655d548b2b8 --hard && \
     pip install --no-cache-dir -e . && \
     echo "✓ Qlib repository cloned and installed successfully") || \
    echo "⚠️ WARNING: Failed to clone qlib repository, fin_quant will build its own image at runtime"

# ==============================================================================
# Python dependencies - RL scenario (MiniWoB, GSM8K, EvalPlus)
# ==============================================================================
RUN pip install --no-cache-dir \
    torch_geometric \
    pytorch_lightning \
    ogb \
    networkx \
    scikit-learn \
    sparse \
    lightgbm==3.3.5 \
    pyarrow \
    fastparquet \
    optuna

# ==============================================================================
# Python dependencies - MLE-Bench / Kaggle scenario
# ==============================================================================
RUN conda init bash && \
    conda create -n mlebench python=3.11 pip -y

RUN set -eux; \
    cd /workspace 2>/dev/null || mkdir -p /workspace && cd /workspace && \
    (git clone --depth 1 https://github.com/openai/mle-bench.git && \
     cd mle-bench && \
     git lfs fetch --all && \
     git lfs pull && \
     conda run -n mlebench pip install -e . && \
     echo "✓ MLE-Bench installed successfully") || \
    echo "⚠️ WARNING: MLE-Bench clone failed, will install on demand"

# ==============================================================================
# Python dependencies - finetune scenario (LLM post-training)
# ==============================================================================
RUN pip install --no-cache-dir \
    trl==0.27.0 \
    peft \
    verl==0.7.0 \
    litellm>=1.73 \
    "transformers>=4.50,<5.0" \
    "bitsandbytes>=0.39.0" \
    "mixture-of-depth>=1.1.6"

# ==============================================================================
# Install RD-Agent in development mode
# ==============================================================================
COPY pyproject.toml .
COPY README.md .
COPY rdagent/ ./rdagent/

RUN pip install -e .

# ==============================================================================
# Create workspace directories
# ==============================================================================
RUN mkdir -p /app/workspace /app/log /app/git_ignore_folder /workspace

# ==============================================================================
# Environment variables
# ==============================================================================
ENV RDAGENT_WORKSPACE=/app/workspace
ENV RDAGENT_LOG_DIR=/app/log

# ==============================================================================
# Health check
# ==============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import rdagent; print('RD-Agent healthy')" || exit 1

# ==============================================================================
# Default command
# ==============================================================================
ENTRYPOINT ["rdagent"]
CMD ["--help"]

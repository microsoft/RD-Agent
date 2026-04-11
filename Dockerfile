# RD-Agent Dockerfile
# Build: docker build -t rdagent:latest .
# Build with GPU: docker build --build-arg GPU_SUPPORT=true -t rdagent:gpu .
# Run: docker run -it --rm \
#        -v $(pwd)/.env:/app/.env \
#        -v $(pwd)/workspace:/app/workspace \
#        -v /var/run/docker.sock:/var/run/docker.sock \
#        rdagent:latest

# Choose base image based on GPU support
ARG GPU_SUPPORT=false
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS gpu-base
FROM python:3.10-slim AS cpu-base
FROM ${GPU_SUPPORT:+gpu-base}${GPU_SUPPORT:-cpu-base} AS base

FROM base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# NOTE: Proxy handling for China users - automatically bypass proxy for local mirrors
RUN set -eux; \
    for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do \
        proxy_value=$(printenv "$proxy_var" || true); \
        case "$proxy_value" in \
            http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*) \
                unset "$proxy_var"; \
                ;; \
        esac; \
    done; \
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
        # Clean up
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Install Google Chrome for web crawling
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

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker cache
COPY requirements.txt .

# Install Python dependencies with retry for China users
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
    pip install --no-cache-dir -r requirements.txt \
    # Install qlib for fin_factor/fin_quant scenarios (pip package for basic functionality)
    && pip install --no-cache-dir pyqlib \
    # Install catboost, xgboost for ML models
    && pip install --no-cache-dir catboost xgboost \
    # Install tables for HDF5 support
    && pip install --no-cache-dir tables

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
     echo "Qlib repository cloned and installed successfully") || \
    echo "WARNING: Failed to clone qlib repository, basic fin_factor will still work via pyqlib package"

# Install RD-Agent in development mode
COPY pyproject.toml .
COPY README.md .
COPY rdagent/ ./rdagent/

RUN pip install -e .

# Create workspace directory
RUN mkdir -p /app/workspace /app/log /app/git_ignore_folder

# Set environment variables for RD-Agent
ENV RDAGENT_WORKSPACE=/app/workspace
ENV RDAGENT_LOG_DIR=/app/log

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import rdagent; print('RD-Agent healthy')" || exit 1

# Default command
ENTRYPOINT ["rdagent"]
CMD ["--help"]

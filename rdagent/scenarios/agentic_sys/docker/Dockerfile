FROM python:3.11-slim

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    && pip install --no-cache-dir uv \
    && rm -rf /var/lib/apt/lists/*

# Copy entrypoint.sh script into the container workspace  
COPY entrypoint.sh /workspace/entrypoint.sh  
RUN chmod +x /workspace/entrypoint.sh


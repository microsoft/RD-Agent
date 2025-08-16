FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    git-lfs \
    unzip \
    && rm -rf /var/lib/apt/lists/* 

# Install LLaMA-Factory for LLM fine-tuning
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /llamafactory
WORKDIR /llamafactory

# Install LLaMA-Factory with torch and metrics extras
RUN PIP_NO_BUILD_ISOLATION=1 pip install -e ".[torch,metrics]"

# Set working directory for experiments
WORKDIR /workspace/llm_finetune

# Create necessary directories
RUN mkdir -p /workspace/llm_finetune/data /workspace/llm_finetune/output

# Install additional tools for data processing
RUN pip install datasets pandas numpy huggingface-hub

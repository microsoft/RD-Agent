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
WORKDIR /workspace

# Install additional tools for data processing and logging
RUN pip install datasets pandas numpy huggingface-hub pyyaml tensorboard tensorboardX

# Install optional dependencies for advanced fine-tuning features
# - bitsandbytes: for quantization training (QLoRA, etc.)
# - mixture-of-depth: for mixture-of-depths model architecture
RUN pip install "bitsandbytes>=0.39.0" "mixture-of-depth>=1.1.6"

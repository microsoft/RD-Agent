FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags
# Note: Using 'devel' tag instead of 'runtime' to include CUDA development tools required for DeepSpeed compilation

# Set CUDA environment variables for DeepSpeed compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    git-lfs \
    unzip \
    && rm -rf /var/lib/apt/lists/* 

# Install LLaMA-Factory for LLM fine-tuning
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /llamafactory
WORKDIR /llamafactory

# Install LLaMA-Factory with torch and metrics extras
RUN PIP_NO_BUILD_ISOLATION=1 pip install -e ".[torch,metrics,deepspeed]"

# Set working directory for experiments
WORKDIR /workspace

# Install additional tools for data processing and logging
RUN pip install datasets pandas numpy huggingface-hub pyyaml tensorboard tensorboardX

# Install optional dependencies for advanced fine-tuning features
# - bitsandbytes: for quantization training (QLoRA, etc.)
# - mixture-of-depth: for mixture-of-depths model architecture
RUN pip install "bitsandbytes>=0.39.0" "mixture-of-depth>=1.1.6"

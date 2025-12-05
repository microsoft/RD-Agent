FROM hiyouga/llamafactory:0.9.4

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

RUN pip install "bitsandbytes>=0.39.0" "mixture-of-depth>=1.1.6" "openai"

# Set working directory for experiments
WORKDIR /workspace

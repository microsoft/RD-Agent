FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install git+https://github.com/openai/mle-bench.git

WORKDIR /workspace

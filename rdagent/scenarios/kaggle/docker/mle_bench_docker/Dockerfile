FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* 

RUN git clone https://github.com/openai/mle-bench.git
RUN cd mle-bench && git lfs fetch --all
RUN cd mle-bench && git lfs pull
RUN cd mle-bench && python -m pip install -e .

WORKDIR /workspace

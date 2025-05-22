# Use the official PyTorch image as the base image  
FROM pytorch/pytorch:latest
# FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
# torch.__version__ == '2.4.1+cu121' in "gcr.io/kaggle-gpu-images/python"
  
# Install additional tools  
RUN apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \  
    git-lfs \  
    unzip && \  
    rm -rf /var/lib/apt/lists/*  
  
# Default command to keep the container running  
ENTRYPOINT ["/workspace/run/entrypoint.sh"]

RUN conda init bash

# MLE-Bench
RUN conda create -n mlebench python==3.11 pip -y
RUN cd /workspace && git clone https://github.com/openai/mle-bench.git
RUN cd /workspace/mle-bench && git lfs fetch --all
RUN cd /workspace/mle-bench && git lfs pull
RUN cd /workspace/mle-bench && conda run -n mlebench pip install -e .

# Kaggle Environment
COPY ./kaggle_environment.yaml /workspace
RUN cd /workspace && conda env create -f /workspace/kaggle_environment.yaml

# RD-Agent
RUN cd /workspace && git clone https://github.com/microsoft/RD-Agent
RUN cd RD-Agent && git fetch && make dev


# litellm
RUN cd /workspace && mkdir -p litellm-srv
RUN cd /workspace/litellm-srv && curl https://raw.githubusercontent.com/you-n-g/deploy/refs/heads/master/configs/python/litellm.trapi.yaml -o litellm.trapi.yaml
RUN pip install 'litellm[proxy]'
RUN pip install git+https://github.com/you-n-g/litellm@add_mi_cred_pr

run cd /workspace && mkdir -p run
COPY ./entrypoint.sh /workspace/run


WORKDIR /workspace/RD-Agent/

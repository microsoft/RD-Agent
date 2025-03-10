# Use the official PyTorch image as the base image  
FROM pytorch/pytorch:latest
  
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
CMD ["bash"]

RUN conda init bash

RUN conda create -n mlebench python==3.11 pip -y
RUN cd /workspace & git clone https://github.com/openai/mle-bench.git
RUN cd /workspace/mle-bench && git lfs fetch --all
RUN cd /workspace/mle-bench && git lfs pull
RUN cd /workspace/mle-bench && conda run -n mlebench pip install -e .

COPY ./kaggle_environment.yaml /workspace
RUN cd /workspace & conda env create -f /workspace/kaggle_environment.yaml

RUN git clone https://github.com/microsoft/RD-Agent
RUN cd RD-Agent && git fetch && git checkout xuyang1/add_condaenv && git reset  && make dev

WORKDIR /workspace/RD-Agent/

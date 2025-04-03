FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN git clone --depth=100 https://github.com/microsoft/qlib.git

WORKDIR /workspace/qlib

RUN git reset e7a1b5ea1ff4412792eeb606a639e0dde422053a --hard

RUN python -m pip install --upgrade cython
RUN python -m pip install -e .

RUN pip install catboost
RUN pip install xgboost
RUN pip install scipy==1.11.4
RUN pip install tables

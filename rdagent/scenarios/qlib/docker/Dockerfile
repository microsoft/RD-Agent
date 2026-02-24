FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    coreutils \
    && rm -rf /var/lib/apt/lists/* 

RUN git clone https://github.com/microsoft/qlib.git

WORKDIR /workspace/qlib

RUN git fetch && git reset 2fb9380b342556ddb50a4b24e4fe8655d548b2b8 --hard

RUN python -m pip install --upgrade cython
RUN python -m pip install -e .

RUN pip install catboost
RUN pip install xgboost
RUN pip install tables

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

RUN git reset 4d621bff99bb403231e26bb1dddae6e77d2af170 --hard

RUN python -m pip install --upgrade cython
RUN python -m pip install -e .

RUN pip install catboost
RUN pip install xgboost
RUN pip install scipy==1.11.4

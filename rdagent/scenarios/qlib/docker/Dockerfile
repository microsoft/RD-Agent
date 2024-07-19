FROM pytorch/pytorch:latest

# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN git clone https://github.com/microsoft/qlib.git

WORKDIR /workspace/qlib

RUN git reset c9ed050ef034fe6519c14b59f3d207abcb693282 --hard

RUN python -m pip install --upgrade numpy
RUN python -m pip install --upgrade cython
RUN python -m pip install -e .

RUN pip install catboost
RUN pip install xgboost

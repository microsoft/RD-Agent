FROM gcr.io/kaggle-gpu-images/python:latest

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    strace \
    && rm -rf /var/lib/apt/lists/* 
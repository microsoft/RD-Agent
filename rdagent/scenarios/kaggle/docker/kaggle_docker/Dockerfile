FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /workspace

RUN python -m pip install numpy
# RUN python -m pip install --upgrade cython
# RUN python -m pip install -e .

RUN python -m pip install pandas
# RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.3.0%2Bcu121.html
RUN pip install torch_geometric
RUN pip install pytorch_lightning
RUN pip install ogb
RUN pip install networkx
RUN pip install scikit-learn
RUN pip install catboost
RUN pip install xgboost
RUN pip install sparse
RUN pip install lightgbm==3.3.5
RUN pip install pyarrow
RUN pip install fastparquet
RUN pip install optuna
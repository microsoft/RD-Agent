# ALFWorld 评测环境
FROM python:3.10-slim

WORKDIR /workspace

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（评测所需）
RUN pip install --no-cache-dir \
    alfworld \
    textworld \
    flask \
    loguru \
    pyyaml \
    requests

# 下载 ALFWorld 数据
RUN alfworld-download

# rdagent 代码通过 volume 挂载到 /app/rdagent
ENV PYTHONPATH=/app/rdagent
ENV ALFWORLD_DATA=/root/alfworld-download

EXPOSE 5000
CMD ["bash"]

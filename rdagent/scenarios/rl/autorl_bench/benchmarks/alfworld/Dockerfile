# ALFWorld 评测环境
#
# 构建镜像:  docker build -t autorl-bench/alfworld:latest .
# 运行方式:  由 run.py 的 DockerServerContext 自动管理
#
# 数据流:
#   1. Dockerfile 内 pip install alfworld + alfworld-download 下载 2.1GB 游戏数据
#   2. 游戏数据存储在 /root/.cache/alfworld/ (ALFWORLD_DATA)
#   3. git clone 的仓库挂载到 /data/alfworld/（提供 configs/base_config.yaml）
#   4. rdagent 代码挂载到 /app/rdagent/
#   5. workspace 挂载到 /workspace/

FROM python:3.10-slim

WORKDIR /workspace

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
RUN pip install --no-cache-dir \
    alfworld \
    textworld \
    flask \
    loguru \
    pyyaml \
    requests \
    pandas \
    openai \
    vllm || true

# 下载 ALFWorld 游戏数据（json_2.1.1 + detectors + logic，约 2.1GB）
# alfworld-download 默认下载到 ~/.cache/alfworld/
RUN alfworld-download

# 环境变量
ENV ALFWORLD_DATA=/root/.cache/alfworld
ENV ALFWORLD_CONFIG=/data/alfworld/configs/base_config.yaml
ENV PYTHONPATH=/app/rdagent

EXPOSE 5000
CMD ["bash"]

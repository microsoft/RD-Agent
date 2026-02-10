# MiniWoB 训练+评测镜像
FROM autorl-bench/base:latest

WORKDIR /workspace

# 额外安装：浏览器 + selenium + miniwob
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium chromium-driver \
    fonts-liberation \
    libnss3 libxss1 libasound2 libgbm1 \
    libx11-6 libxext6 libxrender1 libxtst6 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    miniwob \
    gymnasium \
    selenium

ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_BIN=/usr/bin/chromedriver

CMD ["bash"]

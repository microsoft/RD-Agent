FROM python:3.11

WORKDIR /app

# 复制当前目录内容到工作目录，忽略.git_ignore_folder
COPY . /app

# 安装依赖
RUN make dev

# 暴露端口
EXPOSE 8080

# 运行 Streamlit 应用
CMD ["streamlit", "run", "rdagent/log/ui/app.py", "--server.port=8080", "--", "--log_dir=./demo_traces"]

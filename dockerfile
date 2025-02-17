FROM python:3.11

WORKDIR /app

COPY . /app

# 安装依赖
RUN make dev

# 暴露端口
EXPOSE 10010

# 运行 Streamlit 应用
CMD ["python", "rdagent/log/server/app.py"]
# CMD ["streamlit", "run", "rdagent/log/ui/app.py", "--server.port=8080", "--", "--log_dir=./demo_traces"]

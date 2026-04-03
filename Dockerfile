FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build:flask


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
COPY --from=frontend-builder /app/git_ignore_folder/static /app/git_ignore_folder/static

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install .

EXPOSE 19899

CMD sh -c 'rdagent server_ui --port "${PORT:-19899}"'

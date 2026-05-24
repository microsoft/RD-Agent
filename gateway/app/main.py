import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import app.brokers.bybit  # noqa: F401 — register broker adapters
from app.config import settings
from app.routers import agent, execution, health, market, research
from app.services.agent_runner import agent_runner


def _bootstrap_repo_path() -> None:
    root = str(settings.repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def create_app() -> FastAPI:
    _bootstrap_repo_path()

    app = FastAPI(
        title="RD-Agent Terminal Gateway",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(market.router, prefix="/api/v1")
    app.include_router(agent.router, prefix="/api/v1")
    app.include_router(research.router, prefix="/api/v1")
    app.include_router(execution.router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "rdagent-gateway", "docs": "/docs"}

    @app.post("/receive")
    async def receive_msgs(request: Request) -> dict[str, str]:
        payload = await request.json()
        agent_runner.ingest_receive_payload(payload)
        return {"status": "success"}

    return app


app = create_app()

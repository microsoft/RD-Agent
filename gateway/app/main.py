from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.brokers.bybit  # noqa: F401 — register broker adapters
from app.config import settings
from app.routers import health, market


def create_app() -> FastAPI:
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

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "rdagent-gateway", "docs": "/docs"}

    return app


app = create_app()

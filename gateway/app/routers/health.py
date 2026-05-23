from fastapi import APIRouter

from app.config import settings
from app.models.market import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from app.brokers.base import BROKER_REGISTRY

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        brokers=sorted(BROKER_REGISTRY.keys()),
        testnet=settings.bybit_testnet,
    )

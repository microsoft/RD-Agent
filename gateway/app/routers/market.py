from fastapi import APIRouter, HTTPException, Query

from app.brokers.base import get_broker
from app.brokers.errors import (
    BrokerNotFoundError,
    BrokerRateLimitError,
    BrokerUpstreamError,
)
from app.models.market import KlinesResponse, SymbolsResponse, Ticker

router = APIRouter(prefix="/market", tags=["market"])


def _resolve_broker(broker_id: str):
    try:
        return get_broker(broker_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _handle_broker_errors(exc: Exception) -> None:
    if isinstance(exc, BrokerNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, BrokerRateLimitError):
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    if isinstance(exc, BrokerUpstreamError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    raise exc


@router.get("/symbols", response_model=SymbolsResponse)
async def list_symbols(
    broker: str = Query(default="bybit"),
    category: str = Query(default="linear"),
) -> SymbolsResponse:
    adapter = _resolve_broker(broker)
    try:
        symbols = await adapter.get_symbols(category=category)
    except Exception as exc:
        _handle_broker_errors(exc)
    return SymbolsResponse(broker=broker, symbols=symbols)


@router.get("/klines", response_model=KlinesResponse)
async def get_klines(
    symbol: str = Query(..., min_length=1),
    interval: str = Query(default="60"),
    limit: int = Query(default=500, ge=1, le=1000),
    broker: str = Query(default="bybit"),
    category: str = Query(default="linear"),
) -> KlinesResponse:
    adapter = _resolve_broker(broker)
    try:
        bars = await adapter.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            category=category,
        )
    except Exception as exc:
        _handle_broker_errors(exc)
    return KlinesResponse(
        broker=broker,
        symbol=symbol,
        interval=interval,
        bars=bars,
    )


@router.get("/ticker", response_model=Ticker)
async def get_ticker(
    symbol: str = Query(..., min_length=1),
    broker: str = Query(default="bybit"),
    category: str = Query(default="linear"),
) -> Ticker:
    adapter = _resolve_broker(broker)
    try:
        return await adapter.get_ticker(symbol=symbol, category=category)
    except Exception as exc:
        _handle_broker_errors(exc)

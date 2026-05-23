import asyncio
from typing import Any

from pybit.unified_trading import HTTP

from app.brokers.base import register_broker
from app.brokers.errors import BrokerNotFoundError, BrokerUpstreamError
from app.config import settings
from app.models.market import OHLCVBar, Symbol, Ticker


@register_broker("bybit")
class BybitAdapter:
    broker_id = "bybit"
    market_type = "crypto"

    def __init__(
        self,
        testnet: bool | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self._testnet = settings.bybit_testnet if testnet is None else testnet
        self._client = HTTP(
            testnet=self._testnet,
            api_key=api_key if api_key is not None else settings.bybit_api_key or None,
            api_secret=api_secret if api_secret is not None else settings.bybit_api_secret or None,
        )

    async def get_symbols(self, category: str = "linear") -> list[Symbol]:
        response = await asyncio.to_thread(
            self._client.get_instruments_info,
            category=category,
        )
        self._ensure_success(response)
        rows = response.get("result", {}).get("list", [])
        symbols: list[Symbol] = []
        for row in rows:
            if row.get("status") != "Trading":
                continue
            symbols.append(
                Symbol(
                    symbol=row["symbol"],
                    baseCoin=row.get("baseCoin", ""),
                    quoteCoin=row.get("quoteCoin", ""),
                    status=row.get("status", ""),
                )
            )
        return symbols

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        category: str = "linear",
    ) -> list[OHLCVBar]:
        capped_limit = max(1, min(limit, 1000))
        response = await asyncio.to_thread(
            self._client.get_kline,
            category=category,
            symbol=symbol,
            interval=interval,
            limit=capped_limit,
        )
        self._ensure_success(response, symbol=symbol)
        rows = response.get("result", {}).get("list", [])
        bars = [self._parse_kline(row) for row in rows]
        bars.sort(key=lambda bar: bar.time)
        return bars

    async def get_ticker(self, symbol: str, category: str = "linear") -> Ticker:
        response = await asyncio.to_thread(
            self._client.get_tickers,
            category=category,
            symbol=symbol,
        )
        self._ensure_success(response, symbol=symbol)
        rows = response.get("result", {}).get("list", [])
        if not rows:
            raise BrokerNotFoundError(f"Symbol not found: {symbol}")
        row = rows[0]
        return Ticker(
            symbol=row.get("symbol", symbol),
            lastPrice=float(row.get("lastPrice", 0)),
            price24hPcnt=float(row.get("price24hPcnt", 0)),
            volume24h=float(row.get("volume24h", 0)),
            highPrice24h=float(row.get("highPrice24h", 0)),
            lowPrice24h=float(row.get("lowPrice24h", 0)),
        )

    def _parse_kline(self, row: list[Any]) -> OHLCVBar:
        # Bybit list format: [startTime, open, high, low, close, volume, turnover]
        start_ms = int(row[0])
        return OHLCVBar(
            time=start_ms // 1000,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )

    def _ensure_success(self, response: dict[str, Any], symbol: str | None = None) -> None:
        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg", "Bybit API error")
        if ret_code == 0:
            return
        if ret_code in {10001, 10002, 10003, 10004, 10005, 10006, 10007, 10017}:
            raise BrokerNotFoundError(ret_msg or f"Symbol not found: {symbol}")
        raise BrokerUpstreamError(ret_msg)

from unittest.mock import MagicMock, patch

import pytest

from app.brokers.bybit import BybitAdapter
from app.brokers.errors import BrokerNotFoundError, BrokerUpstreamError


@pytest.fixture
def adapter() -> BybitAdapter:
    with patch("app.brokers.bybit.HTTP"):
        return BybitAdapter(testnet=True, api_key="", api_secret="")


@pytest.mark.asyncio
async def test_get_symbols_filters_trading(adapter: BybitAdapter) -> None:
    adapter._client.get_instruments_info = MagicMock(
        return_value={
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "baseCoin": "BTC",
                        "quoteCoin": "USDT",
                        "status": "Trading",
                    },
                    {
                        "symbol": "OLDCOIN",
                        "baseCoin": "OLD",
                        "quoteCoin": "USDT",
                        "status": "Closed",
                    },
                ]
            },
        }
    )
    symbols = await adapter.get_symbols()
    assert len(symbols) == 1
    assert symbols[0].symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_get_klines_sorted_ascending(adapter: BybitAdapter) -> None:
    adapter._client.get_kline = MagicMock(
        return_value={
            "retCode": 0,
            "result": {
                "list": [
                    ["2000000", "2", "3", "1", "2.5", "100", "200"],
                    ["1000000", "1", "2", "0.5", "1.5", "50", "100"],
                ]
            },
        }
    )
    bars = await adapter.get_klines("BTCUSDT", "60", 2)
    assert len(bars) == 2
    assert bars[0].time == 1000
    assert bars[1].time == 2000
    assert bars[0].close == 1.5


@pytest.mark.asyncio
async def test_get_ticker(adapter: BybitAdapter) -> None:
    adapter._client.get_tickers = MagicMock(
        return_value={
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "lastPrice": "65000",
                        "price24hPcnt": "1.23",
                        "volume24h": "999",
                        "highPrice24h": "66000",
                        "lowPrice24h": "64000",
                    }
                ]
            },
        }
    )
    ticker = await adapter.get_ticker("BTCUSDT")
    assert ticker.symbol == "BTCUSDT"
    assert ticker.lastPrice == 65000.0


@pytest.mark.asyncio
async def test_get_ticker_not_found(adapter: BybitAdapter) -> None:
    adapter._client.get_tickers = MagicMock(
        return_value={"retCode": 0, "result": {"list": []}}
    )
    with pytest.raises(BrokerNotFoundError):
        await adapter.get_ticker("INVALID")


@pytest.mark.asyncio
async def test_upstream_error(adapter: BybitAdapter) -> None:
    adapter._client.get_kline = MagicMock(
        return_value={"retCode": 10016, "retMsg": "server error"}
    )
    with pytest.raises(BrokerUpstreamError):
        await adapter.get_klines("BTCUSDT", "60", 10)

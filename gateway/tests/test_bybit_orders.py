from unittest.mock import MagicMock, patch

import pytest

from app.brokers.bybit import BybitAdapter
from app.models.execution import OrderRequest, OrderSide, OrderType


@pytest.fixture
def adapter() -> BybitAdapter:
    with patch("app.brokers.bybit.HTTP"):
        return BybitAdapter(testnet=True, api_key="key", api_secret="secret")


@pytest.mark.asyncio
async def test_place_order_market(adapter: BybitAdapter) -> None:
    adapter._client.place_order = MagicMock(
        return_value={"retCode": 0, "result": {"orderId": "abc123"}}
    )
    order = OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, qty=0.01)
    response = await adapter.place_order(order)
    assert response.order_id == "abc123"
    assert response.mode == "live"
    adapter._client.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_order(adapter: BybitAdapter) -> None:
    adapter._client.cancel_order = MagicMock(return_value={"retCode": 0, "result": {}})
    result = await adapter.cancel_order("BTCUSDT", "abc123")
    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_get_positions(adapter: BybitAdapter) -> None:
    adapter._client.get_positions = MagicMock(
        return_value={
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "size": "0.01",
                        "avgPrice": "65000",
                        "markPrice": "66000",
                        "unrealisedPnl": "10",
                    }
                ]
            },
        }
    )
    positions = await adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSDT"
    assert positions[0].size == 0.01

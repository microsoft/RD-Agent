from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.market import OHLCVBar, Symbol, Ticker


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "bybit" in payload["brokers"]


@patch("app.routers.market.get_broker")
def test_klines(mock_get_broker: AsyncMock, client: TestClient) -> None:
    mock_adapter = AsyncMock()
    mock_adapter.get_klines.return_value = [
        OHLCVBar(time=1000, open=1, high=2, low=0.5, close=1.5, volume=10)
    ]
    mock_get_broker.return_value = mock_adapter

    response = client.get(
        "/api/v1/market/klines",
        params={"symbol": "BTCUSDT", "interval": "60", "limit": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BTCUSDT"
    assert len(payload["bars"]) == 1
    assert payload["bars"][0]["time"] == 1000


@patch("app.routers.market.get_broker")
def test_ticker_not_found(mock_get_broker: AsyncMock, client: TestClient) -> None:
    from app.brokers.errors import BrokerNotFoundError

    mock_adapter = AsyncMock()
    mock_adapter.get_ticker.side_effect = BrokerNotFoundError("missing")
    mock_get_broker.return_value = mock_adapter

    response = client.get("/api/v1/market/ticker", params={"symbol": "BAD"})
    assert response.status_code == 404

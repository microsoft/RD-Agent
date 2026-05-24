from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.execution import OrderSide, OrderType
from app.services.execution_service import execution_service


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_execution_state() -> None:
    execution_service.paper._positions.clear()
    execution_service.paper._orders.clear()
    execution_service.risk.deactivate_kill_switch()
    yield
    execution_service.paper._positions.clear()
    execution_service.paper._orders.clear()
    execution_service.risk.deactivate_kill_switch()


def test_execution_status(client: TestClient) -> None:
    response = client.get("/api/v1/execution/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "paper"
    assert "limits" in payload


@patch("app.services.execution_service.get_broker")
def test_paper_order_success(mock_get_broker, client: TestClient) -> None:
    mock_broker = AsyncMock()
    mock_broker.get_ticker.return_value = type("T", (), {"lastPrice": 65000.0})()
    mock_get_broker.return_value = mock_broker

    response = client.post(
        "/api/v1/execution/orders",
        json={
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY.value,
            "order_type": OrderType.MARKET.value,
            "qty": 0.001,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "paper"
    assert payload["status"] == "Filled"


@patch("app.services.execution_service.get_broker")
def test_paper_order_risk_rejection(mock_get_broker, client: TestClient) -> None:
    mock_broker = AsyncMock()
    mock_broker.get_ticker.return_value = type("T", (), {"lastPrice": 65000.0})()
    mock_get_broker.return_value = mock_broker
    execution_service.risk.max_order_notional = 10

    response = client.post(
        "/api/v1/execution/orders",
        json={
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY.value,
            "order_type": OrderType.MARKET.value,
            "qty": 1,
        },
    )
    assert response.status_code == 422
    execution_service.risk.max_order_notional = 1000


@patch("app.services.execution_service.get_broker")
def test_get_positions_after_order(mock_get_broker, client: TestClient) -> None:
    mock_broker = AsyncMock()
    mock_broker.get_ticker.return_value = type("T", (), {"lastPrice": 65000.0})()
    mock_get_broker.return_value = mock_broker

    client.post(
        "/api/v1/execution/orders",
        json={
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY.value,
            "order_type": OrderType.MARKET.value,
            "qty": 0.001,
        },
    )
    response = client.get("/api/v1/execution/positions")
    assert response.status_code == 200
    assert len(response.json()) == 1

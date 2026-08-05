import pytest

from app.models.execution import OrderRequest, OrderSide, OrderType, Position
from app.services.risk_manager import RiskManager


def _order(qty: float = 0.01, side: OrderSide = OrderSide.BUY) -> OrderRequest:
    return OrderRequest(symbol="BTCUSDT", side=side, order_type=OrderType.MARKET, qty=qty)


def test_check_order_allows_small_order() -> None:
    rm = RiskManager(max_order_notional=1000, max_position_usd=5000, daily_loss_limit=500)
    result = rm.check_order(_order(qty=0.001), mark_price=65000, positions=[])
    assert result.allowed is True


def test_check_order_blocks_oversize_notional() -> None:
    rm = RiskManager(max_order_notional=100, max_position_usd=5000, daily_loss_limit=500)
    result = rm.check_order(_order(qty=1), mark_price=65000, positions=[])
    assert result.allowed is False
    assert any("notional" in r.lower() for r in result.reasons)


def test_check_order_blocks_position_limit() -> None:
    rm = RiskManager(max_order_notional=10000, max_position_usd=1000, daily_loss_limit=500)
    positions = [
        Position(
            symbol="BTCUSDT",
            side="Buy",
            size=0.01,
            avg_price=65000,
            mark_price=65000,
            unrealized_pnl=0,
            notional_usd=650,
        )
    ]
    result = rm.check_order(_order(qty=0.01), mark_price=65000, positions=positions)
    assert result.allowed is False


def test_kill_switch_blocks_orders() -> None:
    rm = RiskManager()
    rm.activate_kill_switch("test")
    result = rm.check_order(_order(), mark_price=65000, positions=[])
    assert result.allowed is False
    assert result.reasons[0] == "test"


def test_daily_loss_triggers_kill_switch() -> None:
    rm = RiskManager(daily_loss_limit=100)
    rm.record_realized_pnl(-150)
    assert rm.is_kill_switch_active() is True

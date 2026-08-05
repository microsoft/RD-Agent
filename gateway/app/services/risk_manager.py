"""Risk management for Phase 3 execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.config import settings
from app.models.execution import OrderRequest, Position, RiskCheckResult


@dataclass
class RiskManager:
    max_order_notional: float = field(default_factory=lambda: settings.max_order_notional)
    max_position_usd: float = field(default_factory=lambda: settings.max_position_usd)
    daily_loss_limit: float = field(default_factory=lambda: settings.daily_loss_limit)
    _kill_switch_active: bool = False
    _kill_switch_reason: str = ""
    _daily_pnl: float = 0.0
    _daily_pnl_date: date = field(default_factory=date.today)

    def _roll_daily_pnl(self) -> None:
        today = date.today()
        if today != self._daily_pnl_date:
            self._daily_pnl = 0.0
            self._daily_pnl_date = today

    def record_realized_pnl(self, amount: float) -> None:
        self._roll_daily_pnl()
        self._daily_pnl += amount
        if self._daily_pnl <= -self.daily_loss_limit:
            self.activate_kill_switch(f"Daily loss limit reached ({self._daily_pnl:.2f})")

    @property
    def daily_pnl(self) -> float:
        self._roll_daily_pnl()
        return self._daily_pnl

    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

    def activate_kill_switch(self, reason: str) -> None:
        self._kill_switch_active = True
        self._kill_switch_reason = reason

    def deactivate_kill_switch(self) -> None:
        self._kill_switch_active = False
        self._kill_switch_reason = ""

    def kill_switch_reason(self) -> str:
        return self._kill_switch_reason

    def check_order(
        self,
        order: OrderRequest,
        mark_price: float,
        positions: list[Position],
    ) -> RiskCheckResult:
        reasons: list[str] = []

        if self.is_kill_switch_active():
            reasons.append(self._kill_switch_reason or "Kill switch is active")

        if order.order_type.value == "Limit" and order.price is None:
            reasons.append("Limit orders require price")

        ref_price = order.price if order.order_type.value == "Limit" and order.price else mark_price
        if ref_price <= 0:
            reasons.append("Invalid reference price for risk check")

        notional = order.qty * ref_price
        if notional > self.max_order_notional:
            reasons.append(
                f"Order notional ${notional:.2f} exceeds max ${self.max_order_notional:.2f}"
            )

        position = next((p for p in positions if p.symbol == order.symbol), None)
        current_notional = position.notional_usd if position else 0.0
        delta = notional if order.side.value == "Buy" else -notional
        projected = abs(current_notional + delta)
        if projected > self.max_position_usd:
            reasons.append(
                f"Projected position ${projected:.2f} exceeds max ${self.max_position_usd:.2f}"
            )

        self._roll_daily_pnl()
        if self._daily_pnl <= -self.daily_loss_limit:
            reasons.append(f"Daily loss limit reached ({self._daily_pnl:.2f})")

        return RiskCheckResult(allowed=len(reasons) == 0, reasons=reasons)

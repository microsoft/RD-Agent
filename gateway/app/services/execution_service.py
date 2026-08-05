"""Order routing: risk checks then paper or live Bybit execution."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from app.brokers.base import get_broker
from app.brokers.bybit import BybitAdapter
from app.config import settings
from app.models.execution import OrderRequest, OrderResponse, PnLSnapshot, Position
from app.services.paper_adapter import PaperAdapter
from app.services.risk_manager import RiskManager


class ExecutionService:
    def __init__(self) -> None:
        self.risk = RiskManager()
        self.paper = PaperAdapter()
        self._pnl_queues: set[asyncio.Queue[PnLSnapshot]] = set()

    @property
    def mode(self) -> str:
        return settings.execution_mode

    async def _mark_prices(self, symbols: list[str], category: str = "linear") -> dict[str, float]:
        broker = get_broker("bybit")
        prices: dict[str, float] = {}
        for symbol in symbols:
            ticker = await broker.get_ticker(symbol=symbol, category=category)
            prices[symbol] = ticker.lastPrice
        return prices

    async def _current_positions(self, category: str = "linear") -> list[Position]:
        if self.mode == "paper":
            symbols = list(self.paper._positions.keys())
            if not symbols:
                return []
            marks = await self._mark_prices(symbols, category=category)
            return await self.paper.get_positions(marks)

        broker = get_broker("bybit")
        if not isinstance(broker, BybitAdapter):
            return []
        return await broker.get_positions(category=category)

    async def submit_order(self, order: OrderRequest) -> OrderResponse:
        if settings.bybit_testnet is False and self.mode == "live":
            raise HTTPException(status_code=403, detail="Live mainnet orders are disabled in Phase 3")

        broker = get_broker(order.broker)
        ticker = await broker.get_ticker(symbol=order.symbol, category=order.category)
        mark_price = ticker.lastPrice
        positions = await self._current_positions(category=order.category)

        risk = self.risk.check_order(order, mark_price=mark_price, positions=positions)
        if not risk.allowed:
            raise HTTPException(status_code=422, detail={"reasons": risk.reasons})

        if self.mode == "paper":
            response = await self.paper.place_order(order, mark_price=mark_price)
        else:
            if not isinstance(broker, BybitAdapter):
                raise HTTPException(status_code=400, detail="Live execution supports Bybit only")
            response = await broker.place_order(order)
            if response.fill_price:
                realized = 0.0
                self.risk.record_realized_pnl(realized)

        await self._broadcast_pnl(category=order.category)
        return response

    async def cancel_order(self, symbol: str, order_id: str, category: str = "linear") -> dict[str, str]:
        if self.mode == "paper":
            if order_id not in self.paper._orders:
                raise HTTPException(status_code=404, detail="Order not found")
            return {"status": "cancelled", "order_id": order_id}

        broker = get_broker("bybit")
        if not isinstance(broker, BybitAdapter):
            raise HTTPException(status_code=400, detail="Cancel supports Bybit only")
        await broker.cancel_order(symbol=symbol, order_id=order_id, category=category)
        return {"status": "cancelled", "order_id": order_id}

    async def get_positions(self, category: str = "linear") -> list[Position]:
        return await self._current_positions(category=category)

    async def get_pnl_snapshot(self, category: str = "linear") -> PnLSnapshot:
        positions = await self._current_positions(category=category)
        unrealized = sum(p.unrealized_pnl for p in positions)
        if self.mode == "paper":
            realized = self.paper.total_realized_pnl()
        else:
            realized = 0.0

        return PnLSnapshot(
            mode=self.mode,
            total_unrealized_pnl=unrealized,
            total_realized_pnl=realized,
            daily_pnl=self.risk.daily_pnl,
            kill_switch_active=self.risk.is_kill_switch_active(),
            positions=positions,
        )

    def subscribe_pnl(self) -> asyncio.Queue[PnLSnapshot]:
        queue: asyncio.Queue[PnLSnapshot] = asyncio.Queue(maxsize=8)
        self._pnl_queues.add(queue)
        return queue

    def unsubscribe_pnl(self, queue: asyncio.Queue[PnLSnapshot]) -> None:
        self._pnl_queues.discard(queue)

    async def _broadcast_pnl(self, category: str = "linear") -> None:
        snapshot = await self.get_pnl_snapshot(category=category)
        dead: list[asyncio.Queue[PnLSnapshot]] = []
        for queue in self._pnl_queues:
            try:
                queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    dead.append(queue)
        for queue in dead:
            self.unsubscribe_pnl(queue)

    async def get_status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "kill_switch_active": self.risk.is_kill_switch_active(),
            "kill_switch_reason": self.risk.kill_switch_reason(),
            "daily_pnl": self.risk.daily_pnl,
            "limits": {
                "max_order_notional": self.risk.max_order_notional,
                "max_position_usd": self.risk.max_position_usd,
                "daily_loss_limit": self.risk.daily_loss_limit,
            },
        }


execution_service = ExecutionService()

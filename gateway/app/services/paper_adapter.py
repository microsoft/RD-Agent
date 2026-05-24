"""In-memory paper trading adapter."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.models.execution import OrderRequest, OrderResponse, OrderSide, OrderType, Position


@dataclass
class _PaperPosition:
    size: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class PaperAdapter:
    slippage_bps: float = 5.0
    _positions: dict[str, _PaperPosition] = field(default_factory=dict)
    _orders: dict[str, OrderResponse] = field(default_factory=dict)

    def _apply_slippage(self, side: OrderSide, price: float) -> float:
        slip = price * (self.slippage_bps / 10_000)
        return price + slip if side == OrderSide.BUY else price - slip

    async def place_order(self, order: OrderRequest, mark_price: float) -> OrderResponse:
        fill_price = order.price if order.order_type == OrderType.LIMIT and order.price else mark_price
        fill_price = self._apply_slippage(order.side, fill_price)
        order_id = f"paper-{uuid.uuid4().hex[:12]}"

        pos = self._positions.setdefault(order.symbol, _PaperPosition())
        signed_qty = order.qty if order.side == OrderSide.BUY else -order.qty
        new_size = pos.size + signed_qty

        if pos.size == 0:
            pos.size = new_size
            pos.avg_price = fill_price
        elif (pos.size > 0 and signed_qty > 0) or (pos.size < 0 and signed_qty < 0):
            total_cost = abs(pos.size) * pos.avg_price + abs(signed_qty) * fill_price
            pos.size = new_size
            pos.avg_price = total_cost / abs(new_size)
        else:
            closed = min(abs(pos.size), abs(signed_qty))
            if pos.size > 0:
                pos.realized_pnl += closed * (fill_price - pos.avg_price)
            else:
                pos.realized_pnl += closed * (pos.avg_price - fill_price)

            if abs(signed_qty) <= abs(pos.size):
                pos.size = new_size
                if abs(pos.size) < 1e-12:
                    pos.avg_price = 0.0
            else:
                pos.size = new_size
                pos.avg_price = fill_price

        response = OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            qty=order.qty,
            price=order.price,
            fill_price=fill_price,
            status="Filled",
            mode="paper",
        )
        self._orders[order_id] = response
        return response

    async def get_positions(self, mark_prices: dict[str, float]) -> list[Position]:
        positions: list[Position] = []
        for symbol, pos in self._positions.items():
            if abs(pos.size) < 1e-12:
                continue
            mark = mark_prices.get(symbol, pos.avg_price)
            side = "Buy" if pos.size > 0 else "Sell"
            size = abs(pos.size)
            if pos.size > 0:
                upnl = size * (mark - pos.avg_price)
            else:
                upnl = size * (pos.avg_price - mark)
            positions.append(
                Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    avg_price=pos.avg_price,
                    mark_price=mark,
                    unrealized_pnl=upnl,
                    notional_usd=size * mark,
                )
            )
        return positions

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

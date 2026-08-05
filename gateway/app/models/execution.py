from enum import Enum

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"


class OrderRequest(BaseModel):
    symbol: str = Field(min_length=1)
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    qty: float = Field(gt=0)
    price: float | None = Field(default=None, gt=0)
    category: str = "linear"
    broker: str = "bybit"


class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    price: float | None
    fill_price: float | None = None
    status: str
    mode: str


class Position(BaseModel):
    symbol: str
    side: str
    size: float
    avg_price: float
    mark_price: float
    unrealized_pnl: float
    notional_usd: float


class RiskCheckResult(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


class PnLSnapshot(BaseModel):
    mode: str
    total_unrealized_pnl: float
    total_realized_pnl: float
    daily_pnl: float
    kill_switch_active: bool
    positions: list[Position]

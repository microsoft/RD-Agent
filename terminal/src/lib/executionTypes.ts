export type OrderSide = "Buy" | "Sell";
export type OrderType = "Market" | "Limit";

export interface OrderRequest {
  symbol: string;
  side: OrderSide;
  order_type: OrderType;
  qty: number;
  price?: number;
  category?: string;
  broker?: string;
}

export interface OrderResponse {
  order_id: string;
  symbol: string;
  side: OrderSide;
  order_type: OrderType;
  qty: number;
  price: number | null;
  fill_price: number | null;
  status: string;
  mode: string;
}

export interface Position {
  symbol: string;
  side: string;
  size: number;
  avg_price: number;
  mark_price: number;
  unrealized_pnl: number;
  notional_usd: number;
}

export interface PnLSnapshot {
  mode: string;
  total_unrealized_pnl: number;
  total_realized_pnl: number;
  daily_pnl: number;
  kill_switch_active: boolean;
  positions: Position[];
}

export interface ExecutionStatus {
  mode: string;
  kill_switch_active: boolean;
  kill_switch_reason: string;
  daily_pnl: number;
  limits: {
    max_order_notional: number;
    max_position_usd: number;
    daily_loss_limit: number;
  };
}

export interface ExecutionPrefill {
  symbol: string;
  side: OrderSide;
  sourceTraceId?: string;
  note?: string;
}

export type CommandCenterTab = "market" | "agent" | "research" | "execution";

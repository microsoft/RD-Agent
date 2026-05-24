import { ApiError } from "@/lib/types";
import type {
  ExecutionStatus,
  OrderRequest,
  OrderResponse,
  PnLSnapshot,
  Position,
} from "@/lib/executionTypes";

const BASE_URL = import.meta.env.VITE_GATEWAY_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = (await response.json()) as Record<string, unknown>;
      if (typeof payload.detail === "string") message = payload.detail;
      else if (payload.detail && typeof payload.detail === "object") {
        const detail = payload.detail as { reasons?: string[] };
        if (detail.reasons?.length) message = detail.reasons.join("; ");
      }
    } catch {
      // ignore parse errors
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export function fetchExecutionStatus(): Promise<ExecutionStatus> {
  return request<ExecutionStatus>("/api/v1/execution/status");
}

export function fetchPositions(category = "linear"): Promise<Position[]> {
  const params = new URLSearchParams({ category });
  return request<Position[]>(`/api/v1/execution/positions?${params.toString()}`);
}

export function fetchPnL(category = "linear"): Promise<PnLSnapshot> {
  const params = new URLSearchParams({ category });
  return request<PnLSnapshot>(`/api/v1/execution/pnl?${params.toString()}`);
}

export function submitOrder(body: OrderRequest): Promise<OrderResponse> {
  return request<OrderResponse>("/api/v1/execution/orders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function activateKillSwitch(reason = "Manual kill switch"): Promise<{ status: string }> {
  const params = new URLSearchParams({ reason });
  return request(`/api/v1/execution/kill-switch/activate?${params.toString()}`, { method: "POST" });
}

export function deactivateKillSwitch(): Promise<{ status: string }> {
  return request("/api/v1/execution/kill-switch/deactivate", { method: "POST" });
}

export function pnlWebSocketUrl(category = "linear"): string {
  const base = BASE_URL || window.location.origin;
  const wsBase = base.replace(/^http/, "ws");
  const params = new URLSearchParams({ category });
  return `${wsBase}/api/v1/execution/ws/pnl?${params.toString()}`;
}

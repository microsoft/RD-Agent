import { ApiError, type HealthResponse, type KlinesResponse, type SymbolsResponse, type Ticker } from "@/lib/types";

const BASE_URL = import.meta.env.VITE_GATEWAY_URL ?? "";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = (await response.json()) as Record<string, unknown>;
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/v1/health");
}

export function fetchSymbols(broker = "bybit", category = "linear"): Promise<SymbolsResponse> {
  const params = new URLSearchParams({ broker, category });
  return request<SymbolsResponse>(`/api/v1/market/symbols?${params.toString()}`);
}

export function fetchKlines(
  symbol: string,
  interval: string,
  limit = 500,
  broker = "bybit",
  category = "linear",
): Promise<KlinesResponse> {
  const params = new URLSearchParams({
    broker,
    category,
    symbol,
    interval,
    limit: String(limit),
  });
  return request<KlinesResponse>(`/api/v1/market/klines?${params.toString()}`);
}

export function fetchTicker(symbol: string, broker = "bybit", category = "linear"): Promise<Ticker> {
  const params = new URLSearchParams({ broker, category, symbol });
  return request<Ticker>(`/api/v1/market/ticker?${params.toString()}`);
}

import { useQuery } from "@tanstack/react-query";
import { fetchHealth, fetchKlines, fetchSymbols, fetchTicker } from "@/lib/api";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
  });
}

export function useSymbols(broker = "bybit") {
  return useQuery({
    queryKey: ["symbols", broker],
    queryFn: () => fetchSymbols(broker),
    staleTime: 60 * 60 * 1000,
  });
}

export function useKlines(symbol: string, interval: string, limit = 500, broker = "bybit") {
  return useQuery({
    queryKey: ["klines", broker, symbol, interval, limit],
    queryFn: () => fetchKlines(symbol, interval, limit, broker),
    enabled: Boolean(symbol && interval),
  });
}

export function useTicker(symbol: string, broker = "bybit") {
  return useQuery({
    queryKey: ["ticker", broker, symbol],
    queryFn: () => fetchTicker(symbol, broker),
    enabled: Boolean(symbol),
    staleTime: 5_000,
    refetchInterval: 5_000,
  });
}

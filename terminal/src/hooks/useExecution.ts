import {
  activateKillSwitch,
  deactivateKillSwitch,
  fetchExecutionStatus,
  fetchPnL,
  fetchPositions,
  pnlWebSocketUrl,
  submitOrder,
} from "@/lib/executionApi";
import type { OrderRequest, PnLSnapshot } from "@/lib/executionTypes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

export function useExecutionStatus() {
  return useQuery({
    queryKey: ["execution", "status"],
    queryFn: fetchExecutionStatus,
    refetchInterval: 10_000,
  });
}

export function usePositions() {
  return useQuery({
    queryKey: ["execution", "positions"],
    queryFn: () => fetchPositions(),
    refetchInterval: 5_000,
  });
}

export function usePnL() {
  return useQuery({
    queryKey: ["execution", "pnl"],
    queryFn: () => fetchPnL(),
    refetchInterval: 5_000,
  });
}

export function useSubmitOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: OrderRequest) => submitOrder(body),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["execution"] });
    },
  });
}

export function useKillSwitch() {
  const queryClient = useQueryClient();
  const activate = useMutation({
    mutationFn: (reason?: string) => activateKillSwitch(reason),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["execution"] }),
  });
  const deactivate = useMutation({
    mutationFn: () => deactivateKillSwitch(),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["execution"] }),
  });
  return { activate, deactivate };
}

export function usePnLWebSocket(enabled = true) {
  const [snapshot, setSnapshot] = useState<PnLSnapshot | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    const ws = new WebSocket(pnlWebSocketUrl());
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      try {
        setSnapshot(JSON.parse(event.data as string) as PnLSnapshot);
      } catch {
        // ignore malformed payloads
      }
    };
    return () => ws.close();
  }, [enabled]);

  return { snapshot, connected };
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAgentScenarios,
  fetchAgentTrace,
  fetchAgentTraces,
  runAgentScenario,
  stopAgentTrace,
} from "@/lib/agentApi";
import { useEffect, useRef, useState } from "react";
import type { TraceMessage } from "@/lib/agentTypes";
import { agentTraceWebSocketUrl } from "@/lib/agentApi";

export function useAgentScenarios() {
  return useQuery({ queryKey: ["agent-scenarios"], queryFn: fetchAgentScenarios });
}

export function useAgentTraces() {
  return useQuery({ queryKey: ["agent-traces"], queryFn: fetchAgentTraces, refetchInterval: 30_000 });
}

export function useAgentTrace(traceId: string | null) {
  return useQuery({
    queryKey: ["agent-trace", traceId],
    queryFn: () => fetchAgentTrace(traceId!, 0, 200, true),
    enabled: Boolean(traceId),
    refetchInterval: traceId ? 5_000 : false,
  });
}

export function useRunAgent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: runAgentScenario,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["agent-traces"] });
    },
  });
}

export function useStopAgent() {
  return useMutation({ mutationFn: stopAgentTrace });
}

export function useAgentTraceWebSocket(traceId: string | null) {
  const [messages, setMessages] = useState<TraceMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const seenRef = useRef(0);

  useEffect(() => {
    if (!traceId) return;
    seenRef.current = 0;
    setMessages([]);
    const ws = new WebSocket(agentTraceWebSocketUrl(traceId));

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data as string) as TraceMessage;
      setMessages((prev) => [...prev, msg]);
      seenRef.current += 1;
    };

    return () => ws.close();
  }, [traceId]);

  return { messages, connected };
}

import type { ExperimentSummary, LoopMetrics, ReturnMarker, ReturnPoint, TraceMessage } from "@/lib/agentTypes";
import { ApiError } from "@/lib/types";

const BASE_URL = import.meta.env.VITE_GATEWAY_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = (await response.json()) as Record<string, unknown>;
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // ignore
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export function fetchAgentScenarios() {
  return request<Array<{ name: string; upload: boolean; developer: boolean }>>("/api/v1/agent/scenarios");
}

export function fetchAgentTraces() {
  return request<string[]>("/api/v1/agent/traces");
}

export function fetchAgentTrace(traceId: string, offset = 0, limit = 100, all = false) {
  const params = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
    all: String(all),
  });
  return request<TraceMessage[]>(`/api/v1/agent/trace/${traceId}?${params.toString()}`);
}

export async function runAgentScenario(formData: FormData) {
  const response = await fetch(`${BASE_URL}/api/v1/agent/run`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new ApiError("Failed to start agent run", response.status);
  }
  return (await response.json()) as { id: string };
}

export function stopAgentTrace(traceId: string) {
  return request<{ status: string }>("/api/v1/agent/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: traceId, action: "stop" }),
  });
}

export function fetchExperiments() {
  return request<ExperimentSummary[]>("/api/v1/research/experiments");
}

export function fetchResearchMetrics(traceId: string) {
  return request<{ traceId: string; loops: LoopMetrics[] }>(
    `/api/v1/research/${encodeURIComponent(traceId)}/metrics`,
  );
}

export function fetchResearchReturns(traceId: string, loopId?: number) {
  const query = loopId !== undefined ? `?loop_id=${loopId}` : "";
  return request<{ traceId: string; points: ReturnPoint[]; markers: ReturnMarker[] }>(
    `/api/v1/research/${encodeURIComponent(traceId)}/returns${query}`,
  );
}

export function agentTraceWebSocketUrl(traceId: string) {
  const base = BASE_URL.replace(/^http/, "ws");
  return `${base}/api/v1/agent/ws/trace/${traceId}`;
}

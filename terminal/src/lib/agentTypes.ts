export const AGENT_SCENARIOS = [
  { name: "Finance Data Building", upload: false, defaultLoops: 3, defaultDuration: 6 },
  { name: "Finance Model Implementation", upload: false, defaultLoops: 3, defaultDuration: 6 },
  { name: "Finance Whole Pipeline", upload: false, defaultLoops: 3, defaultDuration: 6 },
  { name: "Finance Data Building (Reports)", upload: true, defaultLoops: 10, defaultDuration: 24 },
  { name: "General Model Implementation", upload: true, defaultLoops: 1, defaultDuration: 24 },
] as const;

export type AgentScenarioName = (typeof AGENT_SCENARIOS)[number]["name"];

export interface TraceMessage {
  tag: string;
  timestamp?: string;
  loop_id?: number;
  content?: Record<string, unknown> | string | unknown[];
  [key: string]: unknown;
}

export interface ExperimentSummary {
  traceId: string;
  scenario: string;
  traceName: string;
  loopCount: number;
  messageCount: number;
  lastTimestamp?: string | null;
}

export interface LoopMetrics {
  loopId: number;
  metrics: Record<string, number | string>;
  hypothesis?: string | null;
  decision?: boolean | null;
}

export interface ReturnPoint {
  time: string;
  bench: number;
  strategy: number;
  excess: number;
}

export interface ReturnMarker {
  time: string;
  type: string;
}

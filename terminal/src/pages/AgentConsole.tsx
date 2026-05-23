import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LoopTimeline } from "@/components/agent/LoopTimeline";
import { AGENT_SCENARIOS, type AgentScenarioName } from "@/lib/agentTypes";
import {
  useAgentScenarios,
  useAgentTrace,
  useAgentTraceWebSocket,
  useAgentTraces,
  useRunAgent,
  useStopAgent,
} from "@/hooks/useAgent";
import { useState } from "react";

export default function AgentConsole() {
  const [scenario, setScenario] = useState<AgentScenarioName>(AGENT_SCENARIOS[0].name);
  const [loops, setLoops] = useState("1");
  const [duration, setDuration] = useState("6");
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);

  useAgentScenarios();
  const tracesQuery = useAgentTraces();
  const runMutation = useRunAgent();
  const stopMutation = useStopAgent();
  const traceQuery = useAgentTrace(activeTraceId);
  const wsTrace = useAgentTraceWebSocket(activeTraceId);

  const messages = wsTrace.messages.length ? wsTrace.messages : traceQuery.data ?? [];

  const onRun = async () => {
    const formData = new FormData();
    formData.append("scenario", scenario);
    formData.append("loops", loops);
    formData.append("all_duration", duration);
    const result = await runMutation.mutateAsync(formData);
    setActiveTraceId(result.id);
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-4">
        <Select value={scenario} onValueChange={(value) => setScenario(value as AgentScenarioName)}>
          <SelectTrigger>
            <SelectValue placeholder="Scenario" />
          </SelectTrigger>
          <SelectContent>
            {AGENT_SCENARIOS.map((item) => (
              <SelectItem key={item.name} value={item.name}>
                {item.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={loops} onValueChange={setLoops}>
          <SelectTrigger>
            <SelectValue placeholder="Loops" />
          </SelectTrigger>
          <SelectContent>
            {["1", "3", "5", "10"].map((value) => (
              <SelectItem key={value} value={value}>
                {value} loops
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={duration} onValueChange={setDuration}>
          <SelectTrigger>
            <SelectValue placeholder="Duration (h)" />
          </SelectTrigger>
          <SelectContent>
            {["1", "6", "12", "24"].map((value) => (
              <SelectItem key={value} value={value}>
                {value}h
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex gap-2">
          <Button onClick={() => void onRun()} disabled={runMutation.isPending}>
            Run Agent
          </Button>
          <Button
            variant="outline"
            onClick={() => activeTraceId && void stopMutation.mutateAsync(activeTraceId)}
            disabled={!activeTraceId || stopMutation.isPending}
          >
            Stop
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-[240px_1fr]">
        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
          <div className="mb-2 text-xs uppercase text-[var(--color-muted)]">Trace History</div>
          <div className="space-y-1">
            {(tracesQuery.data ?? []).map((traceId) => (
              <button
                key={traceId}
                type="button"
                className={`block w-full truncate rounded px-2 py-1 text-left text-xs hover:bg-[#1f2937] ${
                  activeTraceId === traceId ? "bg-[#1f2937] text-white" : "text-[var(--color-muted)]"
                }`}
                onClick={() => setActiveTraceId(traceId)}
              >
                {traceId}
              </button>
            ))}
          </div>
        </div>

        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium">Live Trace {activeTraceId ? `• ${activeTraceId}` : ""}</div>
            <div className="text-xs text-[var(--color-muted)]">
              WS: {wsTrace.connected ? "connected" : "polling/disconnected"}
            </div>
          </div>
          <LoopTimeline messages={messages} />
        </div>
      </div>
    </div>
  );
}

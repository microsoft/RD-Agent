import { Button } from "@/components/ui/button";
import { EquityCurveChart } from "@/components/research/EquityCurveChart";
import { MetricsTable } from "@/components/research/MetricsTable";
import { useExperiments, useResearchMetrics, useResearchReturns } from "@/hooks/useResearch";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useState } from "react";

function inferSideFromReturns(points: { excess: number }[]): "Buy" | "Sell" {
  if (points.length < 2) return "Buy";
  const last = points[points.length - 1]?.excess ?? 0;
  const prev = points[points.length - 2]?.excess ?? 0;
  return last >= prev ? "Buy" : "Sell";
}

export default function ResearchLab() {
  const experimentsQuery = useExperiments();
  const [traceId, setTraceId] = useState<string | null>(null);
  const metricsQuery = useResearchMetrics(traceId);
  const returnsQuery = useResearchReturns(traceId);
  const { activeSymbol, setActiveTab, setExecutionPrefill } = useWorkspaceStore();

  const handleUseAsSignal = () => {
    const points = returnsQuery.data?.points ?? [];
    const side = inferSideFromReturns(points);
    setExecutionPrefill({
      symbol: activeSymbol,
      side,
      sourceTraceId: traceId ?? undefined,
      note: `Research signal from trace ${traceId ?? "unknown"} — ${side} hint on ${activeSymbol}. Confirm manually in Execution Desk.`,
    });
    setActiveTab("execution");
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
        <div className="mb-2 text-xs uppercase text-[var(--color-muted)]">Experiments</div>
        <div className="space-y-1">
          {(experimentsQuery.data ?? []).map((exp) => (
            <button
              key={exp.traceId}
              type="button"
              onClick={() => setTraceId(exp.traceId)}
              className={`block w-full rounded px-2 py-2 text-left text-xs hover:bg-[#1f2937] ${
                traceId === exp.traceId ? "bg-[#1f2937] text-white" : "text-[var(--color-muted)]"
              }`}
            >
              <div className="truncate font-medium">{exp.traceName}</div>
              <div className="truncate">{exp.scenario}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="text-sm font-medium">Qlib Metrics</div>
            {traceId ? (
              <Button size="sm" variant="outline" onClick={handleUseAsSignal}>
                Use as signal → Execution
              </Button>
            ) : null}
          </div>
          <MetricsTable loops={metricsQuery.data?.loops ?? []} />
        </div>
        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
          <div className="mb-3 text-sm font-medium">Equity Curve</div>
          <EquityCurveChart points={returnsQuery.data?.points ?? []} markers={returnsQuery.data?.markers ?? []} />
        </div>
      </div>
    </div>
  );
}

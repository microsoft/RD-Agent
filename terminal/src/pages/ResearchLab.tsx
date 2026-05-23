import { EquityCurveChart } from "@/components/research/EquityCurveChart";
import { MetricsTable } from "@/components/research/MetricsTable";
import { useExperiments, useResearchMetrics, useResearchReturns } from "@/hooks/useResearch";
import { useState } from "react";

export default function ResearchLab() {
  const experimentsQuery = useExperiments();
  const [traceId, setTraceId] = useState<string | null>(null);
  const metricsQuery = useResearchMetrics(traceId);
  const returnsQuery = useResearchReturns(traceId);

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
          <div className="mb-3 text-sm font-medium">Qlib Metrics</div>
          <MetricsTable loops={metricsQuery.data?.loops ?? []} />
        </div>
        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
          <div className="mb-3 text-sm font-medium">Equity Curve</div>
          <EquityCurveChart points={returnsQuery.data?.points ?? []} />
        </div>
      </div>
    </div>
  );
}

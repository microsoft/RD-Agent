import type { LoopMetrics } from "@/lib/agentTypes";
import { formatPercent, formatPrice } from "@/lib/format";

interface MetricsTableProps {
  loops: LoopMetrics[];
}

const METRIC_KEYS = [
  "IC",
  "ICIR",
  "Rank IC",
  "Rank ICIR",
  "1day.excess_return_with_cost.annualized_return",
  "1day.excess_return_with_cost.max_drawdown",
];

export function MetricsTable({ loops }: MetricsTableProps) {
  if (!loops.length) {
    return <div className="text-sm text-[var(--color-muted)]">No qlib metrics found in trace.</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="text-left text-[var(--color-muted)]">
          <tr>
            <th className="px-2 py-2">Loop</th>
            <th className="px-2 py-2">Decision</th>
            {METRIC_KEYS.map((key) => (
              <th key={key} className="px-2 py-2">
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loops.map((loop) => (
            <tr key={loop.loopId} className="border-t border-[var(--color-border)]">
              <td className="px-2 py-2 font-mono tabular-nums">{loop.loopId}</td>
              <td className="px-2 py-2">{loop.decision === true ? "✓" : loop.decision === false ? "✗" : "—"}</td>
              {METRIC_KEYS.map((key) => {
                const value = loop.metrics[key];
                const display =
                  typeof value === "number"
                    ? key.toLowerCase().includes("drawdown") || key.includes("return")
                      ? formatPercent(value * (key.includes("return") && Math.abs(value) < 2 ? 100 : 1))
                      : formatPrice(value, 4)
                    : value ?? "—";
                return (
                  <td key={key} className="px-2 py-2 font-mono tabular-nums">
                    {display}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

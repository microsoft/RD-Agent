import type { ReturnMarker, ReturnPoint } from "@/lib/agentTypes";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface EquityCurveChartProps {
  points: ReturnPoint[];
  markers?: ReturnMarker[];
}

export function EquityCurveChart({ points, markers = [] }: EquityCurveChartProps) {
  if (!points.length) {
    return <div className="text-sm text-[var(--color-muted)]">No equity curve data for this trace.</div>;
  }

  const markerTimes = new Set(markers.map((m) => m.time));
  const markerColor = (type: string) => (type === "rebalance" ? "#f59e0b" : "#22c55e");

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points}>
          <CartesianGrid stroke="#1f2937" />
          <XAxis dataKey="time" hide />
          <YAxis stroke="#9ca3af" />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
          <Legend />
          <Line type="monotone" dataKey="bench" stroke="#64748b" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="strategy" stroke="#22c55e" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="excess" stroke="#f59e0b" dot={false} strokeWidth={2} />
          {points
            .filter((p) => markerTimes.has(p.time))
            .map((p) => {
              const marker = markers.find((m) => m.time === p.time);
              return (
                <ReferenceDot
                  key={`marker-${p.time}`}
                  x={p.time}
                  y={p.strategy}
                  r={5}
                  fill={markerColor(marker?.type ?? "rebalance")}
                  stroke="#fff"
                  strokeWidth={1}
                />
              );
            })}
        </LineChart>
      </ResponsiveContainer>
      {markers.length ? (
        <div className="mt-2 text-xs text-[var(--color-muted)]">
          Markers: {markers.length} rebalance points (amber dots on strategy line)
        </div>
      ) : null}
    </div>
  );
}

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReturnPoint } from "@/lib/agentTypes";

interface EquityCurveChartProps {
  points: ReturnPoint[];
}

export function EquityCurveChart({ points }: EquityCurveChartProps) {
  if (!points.length) {
    return <div className="text-sm text-[var(--color-muted)]">No equity curve data for this trace.</div>;
  }

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
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

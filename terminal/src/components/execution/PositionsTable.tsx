import type { Position } from "@/lib/executionTypes";
import { formatPrice } from "@/lib/format";

interface PositionsTableProps {
  positions: Position[];
}

export function PositionsTable({ positions }: PositionsTableProps) {
  if (!positions.length) {
    return <div className="text-sm text-[var(--color-muted)]">No open positions.</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-[var(--color-muted)]">
          <tr>
            <th className="pb-2">Symbol</th>
            <th className="pb-2">Side</th>
            <th className="pb-2">Size</th>
            <th className="pb-2">Avg</th>
            <th className="pb-2">Mark</th>
            <th className="pb-2">uPnL</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr key={p.symbol} className="border-t border-[var(--color-border)]">
              <td className="py-2 font-mono">{p.symbol}</td>
              <td className="py-2">{p.side}</td>
              <td className="py-2 font-mono tabular-nums">{p.size}</td>
              <td className="py-2 font-mono tabular-nums">{formatPrice(p.avg_price)}</td>
              <td className="py-2 font-mono tabular-nums">{formatPrice(p.mark_price)}</td>
              <td
                className={`py-2 font-mono tabular-nums ${p.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
              >
                {formatPrice(p.unrealized_pnl)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

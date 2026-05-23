import { Badge } from "@/components/ui/badge";
import { formatPercent, formatPrice } from "@/lib/format";
import type { HealthResponse, Ticker } from "@/lib/types";

interface StatusBarProps {
  health?: HealthResponse;
  healthError?: boolean;
  symbol: string;
  ticker?: Ticker;
}

export function StatusBar({ health, healthError, symbol, ticker }: StatusBarProps) {
  const connected = !healthError && health?.status === "ok";

  return (
    <footer className="flex items-center justify-between border-t border-[var(--color-border)] px-4 py-2 text-xs text-[var(--color-muted)]">
      <div className="flex items-center gap-3">
        <span>Gateway: {connected ? "connected" : "disconnected"}</span>
        <Badge variant={health?.testnet ? "accent" : "default"}>
          Bybit {health?.testnet ? "testnet" : "mainnet"}
        </Badge>
        <span className="font-mono tabular-nums">{symbol}</span>
        {ticker ? (
          <span className="font-mono tabular-nums text-white">
            {formatPrice(ticker.lastPrice)}{" "}
            <span className={ticker.price24hPcnt >= 0 ? "text-green-400" : "text-red-400"}>
              {formatPercent(ticker.price24hPcnt)}
            </span>
          </span>
        ) : null}
      </div>
      <span>RD-Agent Terminal v0.2.0</span>
    </footer>
  );
}

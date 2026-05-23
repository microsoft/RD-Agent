import { useState } from "react";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StatusBar } from "@/components/workspace/StatusBar";
import { WorkspaceShell } from "@/components/workspace/WorkspaceShell";
import { useHealth, useKlines, useSymbols, useTicker } from "@/hooks/useMarket";
import { formatPercent, formatPrice, formatVolume } from "@/lib/format";
import { INTERVAL_OPTIONS } from "@/lib/types";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import AgentConsole from "@/pages/AgentConsole";
import ResearchLab from "@/pages/ResearchLab";

const TABS = [
  { id: "market", label: "Market" },
  { id: "agent", label: "Agent Console" },
  { id: "research", label: "Research Lab" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function CommandCenter() {
  const [tab, setTab] = useState<TabId>("market");
  const { activeSymbol, activeInterval, setActiveSymbol, setActiveInterval } = useWorkspaceStore();
  const healthQuery = useHealth();
  const symbolsQuery = useSymbols();
  const klinesQuery = useKlines(activeSymbol, activeInterval);
  const tickerQuery = useTicker(activeSymbol);

  const symbolOptions =
    symbolsQuery.data?.symbols.map((item) => item.symbol) ?? ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

  const klinesError =
    klinesQuery.error instanceof Error ? klinesQuery.error.message : klinesQuery.isError ? "Failed to load klines" : null;

  return (
    <div className="flex min-h-screen flex-col bg-[var(--color-background)]">
      <header className="border-b border-[var(--color-border)] px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">RD-Agent Terminal</h1>
            <p className="text-sm text-[var(--color-muted)]">Market • Agent • Research — Phase 2</p>
          </div>
          {tab === "market" ? (
            <div className="flex items-center gap-3">
              <Select value={activeSymbol} onValueChange={setActiveSymbol}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Symbol" />
                </SelectTrigger>
                <SelectContent>
                  {symbolOptions.map((symbol) => (
                    <SelectItem key={symbol} value={symbol}>
                      {symbol}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={activeInterval} onValueChange={setActiveInterval}>
                <SelectTrigger className="w-[100px]">
                  <SelectValue placeholder="Interval" />
                </SelectTrigger>
                <SelectContent>
                  {INTERVAL_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : null}
        </div>
        <nav className="mt-4 flex gap-2">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setTab(item.id)}
              className={`rounded px-3 py-1.5 text-sm ${
                tab === item.id ? "bg-[var(--color-accent)] text-black" : "bg-[var(--color-surface)] text-[var(--color-muted)]"
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="min-h-0 flex-1">
        {tab === "market" ? (
          <WorkspaceShell
            chart={
              <CandlestickChart
                bars={klinesQuery.data?.bars ?? []}
                loading={klinesQuery.isLoading}
                error={klinesError}
                onRetry={() => void klinesQuery.refetch()}
              />
            }
            ticker={
              tickerQuery.isLoading ? (
                <div className="text-sm text-[var(--color-muted)]">Loading ticker...</div>
              ) : tickerQuery.data ? (
                <dl className="space-y-3 text-sm">
                  <div className="flex justify-between gap-4">
                    <dt className="text-[var(--color-muted)]">Last</dt>
                    <dd className="font-mono tabular-nums">{formatPrice(tickerQuery.data.lastPrice)}</dd>
                  </div>
                  <div className="flex justify-between gap-4">
                    <dt className="text-[var(--color-muted)]">24h Change</dt>
                    <dd
                      className={`font-mono tabular-nums ${
                        tickerQuery.data.price24hPcnt >= 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {formatPercent(tickerQuery.data.price24hPcnt)}
                    </dd>
                  </div>
                  <div className="flex justify-between gap-4">
                    <dt className="text-[var(--color-muted)]">24h High</dt>
                    <dd className="font-mono tabular-nums">{formatPrice(tickerQuery.data.highPrice24h)}</dd>
                  </div>
                  <div className="flex justify-between gap-4">
                    <dt className="text-[var(--color-muted)]">24h Low</dt>
                    <dd className="font-mono tabular-nums">{formatPrice(tickerQuery.data.lowPrice24h)}</dd>
                  </div>
                  <div className="flex justify-between gap-4">
                    <dt className="text-[var(--color-muted)]">24h Volume</dt>
                    <dd className="font-mono tabular-nums">{formatVolume(tickerQuery.data.volume24h)}</dd>
                  </div>
                </dl>
              ) : (
                <div className="text-sm text-red-300">Ticker unavailable</div>
              )
            }
          />
        ) : null}
        {tab === "agent" ? (
          <div className="p-4">
            <AgentConsole />
          </div>
        ) : null}
        {tab === "research" ? (
          <div className="p-4">
            <ResearchLab />
          </div>
        ) : null}
      </main>

      <StatusBar
        health={healthQuery.data}
        healthError={healthQuery.isError}
        symbol={activeSymbol}
        ticker={tickerQuery.data}
      />
    </div>
  );
}

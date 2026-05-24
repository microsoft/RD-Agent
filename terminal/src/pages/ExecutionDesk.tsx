import { OrderForm } from "@/components/execution/OrderForm";
import { PositionsTable } from "@/components/execution/PositionsTable";
import { KillSwitchControls, RiskBanner } from "@/components/execution/RiskBanner";
import {
  useExecutionStatus,
  useKillSwitch,
  usePnL,
  usePnLWebSocket,
  useSubmitOrder,
} from "@/hooks/useExecution";
import { formatPrice } from "@/lib/format";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useState } from "react";

export default function ExecutionDesk() {
  const { activeSymbol, executionPrefill, clearExecutionPrefill } = useWorkspaceStore();
  const statusQuery = useExecutionStatus();
  const pnlQuery = usePnL();
  const submitOrder = useSubmitOrder();
  const killSwitch = useKillSwitch();
  const { snapshot: wsSnapshot, connected } = usePnLWebSocket();
  const [lastError, setLastError] = useState<string | null>(null);
  const [lastOrderId, setLastOrderId] = useState<string | null>(null);

  const pnl = wsSnapshot ?? pnlQuery.data;

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
      <div className="space-y-4">
        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <div className="mb-3 text-sm font-medium">Order Ticket</div>
          <RiskBanner status={statusQuery.data} rejection={lastError} />
          <div className="mt-3">
            <OrderForm
              symbol={activeSymbol}
              prefill={executionPrefill}
              submitting={submitOrder.isPending}
              onSubmit={(payload) => {
                setLastError(null);
                submitOrder.mutate(payload, {
                  onSuccess: (data) => {
                    setLastOrderId(data.order_id);
                    clearExecutionPrefill();
                  },
                  onError: (err) => setLastError(err instanceof Error ? err.message : "Order failed"),
                });
              }}
            />
          </div>
          {lastOrderId ? (
            <div className="mt-3 text-xs text-green-400">Last order: {lastOrderId}</div>
          ) : null}
        </div>

        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium">Risk Controls</div>
            <KillSwitchControls
              active={statusQuery.data?.kill_switch_active ?? false}
              loading={killSwitch.activate.isPending || killSwitch.deactivate.isPending}
              onActivate={() => killSwitch.activate.mutate("Manual kill switch from terminal")}
              onDeactivate={() => killSwitch.deactivate.mutate()}
            />
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium">P&amp;L</div>
            <div className="text-xs text-[var(--color-muted)]">{connected ? "WS live" : "Polling"}</div>
          </div>
          {pnl ? (
            <dl className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
              <div>
                <dt className="text-[var(--color-muted)]">Unrealized</dt>
                <dd className={`font-mono ${pnl.total_unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {formatPrice(pnl.total_unrealized_pnl)}
                </dd>
              </div>
              <div>
                <dt className="text-[var(--color-muted)]">Realized</dt>
                <dd className="font-mono">{formatPrice(pnl.total_realized_pnl)}</dd>
              </div>
              <div>
                <dt className="text-[var(--color-muted)]">Daily</dt>
                <dd className={`font-mono ${pnl.daily_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {formatPrice(pnl.daily_pnl)}
                </dd>
              </div>
              <div>
                <dt className="text-[var(--color-muted)]">Mode</dt>
                <dd className="font-mono uppercase">{pnl.mode}</dd>
              </div>
            </dl>
          ) : (
            <div className="text-sm text-[var(--color-muted)]">Loading P&amp;L...</div>
          )}
        </div>

        <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <div className="mb-3 text-sm font-medium">Positions</div>
          <PositionsTable positions={pnl?.positions ?? []} />
        </div>
      </div>
    </div>
  );
}

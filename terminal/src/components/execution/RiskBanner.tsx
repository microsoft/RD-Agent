import { Button } from "@/components/ui/button";
import type { ExecutionStatus } from "@/lib/executionTypes";

interface RiskBannerProps {
  status?: ExecutionStatus;
  rejection?: string | null;
}

export function RiskBanner({ status, rejection }: RiskBannerProps) {
  if (rejection) {
    return (
      <div className="rounded border border-red-500/40 bg-red-950/40 px-3 py-2 text-sm text-red-200">
        Order blocked: {rejection}
      </div>
    );
  }

  if (status?.kill_switch_active) {
    return (
      <div className="rounded border border-red-500/40 bg-red-950/40 px-3 py-2 text-sm text-red-200">
        Kill switch active — new orders blocked
        {status.kill_switch_reason ? `: ${status.kill_switch_reason}` : ""}
      </div>
    );
  }

  if (!status) return null;

  return (
    <div className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-xs text-[var(--color-muted)]">
      Mode: <span className="text-white">{status.mode}</span> · Max order ${status.limits.max_order_notional} · Max
      position ${status.limits.max_position_usd} · Daily loss limit ${status.limits.daily_loss_limit}
    </div>
  );
}

interface KillSwitchControlsProps {
  active: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
  loading?: boolean;
}

export function KillSwitchControls({ active, onActivate, onDeactivate, loading }: KillSwitchControlsProps) {
  return active ? (
    <Button
      variant="outline"
      size="sm"
      className="border-red-500/50 text-red-300"
      disabled={loading}
      onClick={onDeactivate}
    >
      Deactivate kill switch
    </Button>
  ) : (
    <Button variant="outline" size="sm" disabled={loading} onClick={onActivate}>
      Activate kill switch
    </Button>
  );
}

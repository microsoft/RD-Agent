import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ExecutionPrefill, OrderSide, OrderType } from "@/lib/executionTypes";
import { useEffect, useState } from "react";

interface OrderFormProps {
  symbol: string;
  prefill?: ExecutionPrefill | null;
  onSubmit: (payload: { symbol: string; side: OrderSide; order_type: OrderType; qty: number; price?: number }) => void;
  submitting?: boolean;
}

export function OrderForm({ symbol, prefill, onSubmit, submitting }: OrderFormProps) {
  const [side, setSide] = useState<OrderSide>(prefill?.side ?? "Buy");
  const [orderType, setOrderType] = useState<OrderType>("Market");
  const [qty, setQty] = useState("0.001");
  const [price, setPrice] = useState("");

  useEffect(() => {
    if (prefill?.side) setSide(prefill.side);
  }, [prefill?.side, prefill?.sourceTraceId]);

  return (
    <div className="space-y-3">
      {prefill?.note ? (
        <div className="rounded border border-amber-500/30 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">
          {prefill.note}
        </div>
      ) : null}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="mb-1 block text-xs text-[var(--color-muted)]">Symbol</label>
          <div className="rounded border border-[var(--color-border)] bg-[#111827] px-3 py-2 font-mono text-sm">
            {symbol}
          </div>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[var(--color-muted)]">Side</label>
          <Select value={side} onValueChange={(v) => setSide(v as OrderSide)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Buy">Buy</SelectItem>
              <SelectItem value="Sell">Sell</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[var(--color-muted)]">Type</label>
          <Select value={orderType} onValueChange={(v) => setOrderType(v as OrderType)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Market">Market</SelectItem>
              <SelectItem value="Limit">Limit</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[var(--color-muted)]">Qty</label>
          <input
            className="w-full rounded border border-[var(--color-border)] bg-[#111827] px-3 py-2 font-mono text-sm"
            value={qty}
            onChange={(e) => setQty(e.target.value)}
          />
        </div>
        {orderType === "Limit" ? (
          <div className="col-span-2">
            <label className="mb-1 block text-xs text-[var(--color-muted)]">Limit price</label>
            <input
              className="w-full rounded border border-[var(--color-border)] bg-[#111827] px-3 py-2 font-mono text-sm"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
        ) : null}
      </div>
      <Button
        className="w-full"
        disabled={submitting}
        onClick={() =>
          onSubmit({
            symbol,
            side,
            order_type: orderType,
            qty: Number(qty),
            price: orderType === "Limit" && price ? Number(price) : undefined,
          })
        }
      >
        {submitting ? "Submitting..." : "Submit order (manual approval)"}
      </Button>
    </div>
  );
}

import { Badge } from "@/components/ui/badge";
import type { TraceMessage } from "@/lib/agentTypes";

interface LoopTimelineProps {
  messages: TraceMessage[];
}

export function LoopTimeline({ messages }: LoopTimelineProps) {
  const items = messages.filter((msg) =>
    ["research.hypothesis", "feedback.hypothesis_feedback", "feedback.metric", "END"].includes(msg.tag),
  );

  if (!items.length) {
    return <div className="text-sm text-[var(--color-muted)]">Waiting for agent events...</div>;
  }

  return (
    <div className="space-y-3">
      {items.map((msg, index) => (
        <div key={`${msg.tag}-${msg.timestamp ?? index}`} className="rounded border border-[var(--color-border)] p-3">
          <div className="mb-1 flex items-center gap-2">
            <Badge variant="accent">{msg.tag}</Badge>
            {msg.loop_id !== undefined ? <span className="text-xs text-[var(--color-muted)]">loop {msg.loop_id}</span> : null}
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-[var(--color-muted)]">
            {JSON.stringify(msg.content ?? {}, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  );
}

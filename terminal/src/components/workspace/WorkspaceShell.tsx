import { useMemo } from "react";
import GridLayout, { type Layout } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";
import { Panel } from "@/components/workspace/Panel";
import { useWorkspaceStore } from "@/stores/workspaceStore";

interface WorkspaceShellProps {
  chart: React.ReactNode;
  ticker: React.ReactNode;
}

export function WorkspaceShell({ chart, ticker }: WorkspaceShellProps) {
  const { layout, setLayout } = useWorkspaceStore();

  const onLayoutChange = (nextLayout: Layout[]) => {
    setLayout(nextLayout);
  };

  const children = useMemo(
    () => ({
      chart,
      ticker,
      agent: (
        <div className="flex h-full items-center justify-center text-sm text-[var(--color-muted)]">
          Agent Console — Phase 2
        </div>
      ),
      execution: (
        <div className="flex h-full items-center justify-center text-sm text-[var(--color-muted)]">
          Execution Monitor — Phase 3
        </div>
      ),
    }),
    [chart, ticker],
  );

  return (
    <div className="min-h-0 flex-1 p-4">
      <GridLayout
        className="layout"
        layout={layout}
        cols={12}
        rowHeight={30}
        width={1200}
        draggableHandle=".panel-drag-handle"
        onLayoutChange={onLayoutChange}
      >
        <div key="chart">
          <Panel title="Market Chart">
            <div className="panel-drag-handle mb-2 cursor-move text-[10px] uppercase text-[var(--color-muted)]">
              Drag panel
            </div>
            <div className="h-[360px]">{children.chart}</div>
          </Panel>
        </div>
        <div key="ticker">
          <Panel title="24h Stats">{children.ticker}</Panel>
        </div>
        <div key="agent">
          <Panel title="Agent Console">{children.agent}</Panel>
        </div>
        <div key="execution">
          <Panel title="Execution Monitor">{children.execution}</Panel>
        </div>
      </GridLayout>
    </div>
  );
}

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Layout } from "react-grid-layout";
import type { CommandCenterTab, ExecutionPrefill } from "@/lib/executionTypes";

const DEFAULT_LAYOUT: Layout[] = [
  { i: "chart", x: 0, y: 0, w: 8, h: 12, minW: 4, minH: 8 },
  { i: "ticker", x: 8, y: 0, w: 4, h: 4, minW: 3, minH: 3 },
  { i: "agent", x: 8, y: 4, w: 4, h: 4, minW: 3, minH: 3 },
  { i: "execution", x: 8, y: 8, w: 4, h: 4, minW: 3, minH: 3 },
];

interface WorkspaceState {
  layout: Layout[];
  activeSymbol: string;
  activeInterval: string;
  activeTab: CommandCenterTab;
  executionPrefill: ExecutionPrefill | null;
  setLayout: (layout: Layout[]) => void;
  setActiveSymbol: (symbol: string) => void;
  setActiveInterval: (interval: string) => void;
  setActiveTab: (tab: CommandCenterTab) => void;
  setExecutionPrefill: (prefill: ExecutionPrefill | null) => void;
  clearExecutionPrefill: () => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      layout: DEFAULT_LAYOUT,
      activeSymbol: "BTCUSDT",
      activeInterval: "60",
      activeTab: "market",
      executionPrefill: null,
      setLayout: (layout) => set({ layout }),
      setActiveSymbol: (activeSymbol) => set({ activeSymbol }),
      setActiveInterval: (activeInterval) => set({ activeInterval }),
      setActiveTab: (activeTab) => set({ activeTab }),
      setExecutionPrefill: (executionPrefill) => set({ executionPrefill }),
      clearExecutionPrefill: () => set({ executionPrefill: null }),
    }),
    {
      name: "rdagent-terminal-workspace",
      partialize: (state) => ({
        layout: state.layout,
        activeSymbol: state.activeSymbol,
        activeInterval: state.activeInterval,
        activeTab: state.activeTab,
      }),
    },
  ),
);

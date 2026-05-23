import { useQuery } from "@tanstack/react-query";
import { fetchExperiments, fetchResearchMetrics, fetchResearchReturns } from "@/lib/agentApi";

export function useExperiments() {
  return useQuery({ queryKey: ["research-experiments"], queryFn: fetchExperiments, refetchInterval: 30_000 });
}

export function useResearchMetrics(traceId: string | null) {
  return useQuery({
    queryKey: ["research-metrics", traceId],
    queryFn: () => fetchResearchMetrics(traceId!),
    enabled: Boolean(traceId),
  });
}

export function useResearchReturns(traceId: string | null, loopId?: number) {
  return useQuery({
    queryKey: ["research-returns", traceId, loopId],
    queryFn: () => fetchResearchReturns(traceId!, loopId),
    enabled: Boolean(traceId),
  });
}

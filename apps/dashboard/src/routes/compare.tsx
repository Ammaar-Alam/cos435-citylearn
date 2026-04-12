import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

import { CompareBars, RunsTable, SectionHeader } from "../components";
import { fetchRuns } from "../lib/api";

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: fetchRuns });
  const selectedIds = new Set((searchParams.get("runIds") ?? "").split(",").filter(Boolean));
  const runs = runsQuery.data ?? [];
  const selectedRuns = runs.filter((run) => selectedIds.has(run.run_id));

  function toggle(runId: string): void {
    const next = new Set(selectedIds);
    if (next.has(runId)) {
      next.delete(runId);
    } else {
      next.add(runId);
    }
    setSearchParams({ runIds: Array.from(next).join(",") });
  }

  return (
    <div className="page-stack">
      <section className="panel panel--feature">
        <SectionHeader
          eyebrow="comparison"
          title="Side-by-side benchmark view"
          copy="Use this for quick method checks before the paper figures get regenerated from the canonical CSV and JSON outputs."
        />
        <div className="compare-picker">
          {runs.map((run) => (
            <label key={run.run_id} className={`compare-pill ${selectedIds.has(run.run_id) ? "is-active" : ""}`}>
              <input checked={selectedIds.has(run.run_id)} onChange={() => toggle(run.run_id)} type="checkbox" />
              <span>{run.run_id}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="content-grid">
        <div className="panel">
          <SectionHeader
            eyebrow="score bars"
            title="Average score comparison"
            copy="Lower is better. This is for interactive inspection, not the final claim figure."
          />
          <CompareBars runs={selectedRuns} />
        </div>
        <div className="panel">
          <SectionHeader
            eyebrow="selected runs"
            title="Run ledger"
            copy="Keep the comparison set small enough that the deltas stay interpretable."
          />
          <RunsTable runs={selectedRuns} />
        </div>
      </section>
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { RunsTable, SectionHeader } from "../components";
import { fetchRuns } from "../lib/api";

export function RunsPage() {
  const [query, setQuery] = useState("");
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: fetchRuns });
  const filteredRuns = useMemo(() => {
    const runs = runsQuery.data ?? [];
    if (!query.trim()) {
      return runs;
    }

    const needle = query.trim().toLowerCase();
    return runs.filter((run) =>
      [run.run_id, run.algorithm, run.variant, run.split, run.dataset_name]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [query, runsQuery.data]);

  return (
    <div className="page-stack">
      <section className="panel">
        <SectionHeader
          eyebrow="run archive"
          title="All discovered runs"
          copy="Filter by run id, method, split, or dataset. The detail page gives the playback surface and the richer KPI breakdown."
          action={
            <input
              className="search-input"
              onChange={(event) => setQuery(event.target.value)}
              placeholder="filter runs"
              value={query}
            />
          }
        />
        <RunsTable runs={filteredRuns} />
      </section>
    </div>
  );
}

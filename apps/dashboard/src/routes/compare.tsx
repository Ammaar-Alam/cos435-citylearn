import { useQueries, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { CompareBars, MetricCard, SectionHeader } from "../components";
import { fetchRunDetail, fetchRuns } from "../lib/api";
import { OFFICIAL_BENCHMARK } from "../lib/benchmark";
import {
  formatCompactRunId,
  formatRunContext,
  formatRunTitle,
  formatScore,
  getRunArtifactKinds,
} from "../lib/format";

const MAX_COMPARE = 4;

export function ComparePage() {
  const [search, setSearch] = useState("");
  const [searchParams, setSearchParams] = useSearchParams();
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: fetchRuns });
  const runs = runsQuery.data ?? [];

  const selectedIds = new Set((searchParams.get("runIds") ?? "").split(",").filter(Boolean));
  const selectedRuns = runs.filter((run) => selectedIds.has(run.run_id));
  const filteredRuns = useMemo(() => {
    if (!search.trim()) {
      return runs;
    }

    const needle = search.trim().toLowerCase();
    return runs.filter((run) =>
      [run.run_id, run.algorithm, run.variant, run.split, run.dataset_name]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [runs, search]);

  const detailQueries = useQueries({
    queries: selectedRuns.map((run) => ({
      queryKey: ["run", run.run_id],
      queryFn: () => fetchRunDetail(run.run_id),
    })),
  });
  const selectedDetails = detailQueries
    .map((query) => query.data)
    .filter((detail): detail is NonNullable<(typeof detailQueries)[number]["data"]> => Boolean(detail));
  const metricKeys = Array.from(
    new Set(selectedDetails.flatMap((detail) => Object.keys(detail.challenge_metrics))),
  ).filter((key) => key !== "average_score");
  const bestSelectedRun = selectedRuns.reduce<typeof selectedRuns[number] | null>((best, run) => {
    if (run.average_score === null) {
      return best;
    }
    if (best === null || best.average_score === null) {
      return run;
    }
    return run.average_score < best.average_score ? run : best;
  }, null);

  function updateSelected(next: Set<string>): void {
    setSearchParams(next.size > 0 ? { runIds: Array.from(next).join(",") } : {});
  }

  function toggle(runId: string): void {
    const next = new Set(selectedIds);
    if (next.has(runId)) {
      next.delete(runId);
      updateSelected(next);
      return;
    }

    if (next.size >= MAX_COMPARE) {
      return;
    }

    next.add(runId);
    updateSelected(next);
  }

  return (
    <div className="page-stack page-stack--compare">
      <section className="page-header">
        <div className="page-header__body">
          <div className="page-header__eyebrow">Compare</div>
          <h1>Compare runs</h1>
          <p>Pick 2–4 runs and read the deltas.</p>
        </div>
        <div className="page-header__actions">
          <input
            className="search-input"
            onChange={(event) => setSearch(event.target.value)}
            placeholder="search runs to compare"
            value={search}
          />
          <button className="ghost-button ghost-button--small" onClick={() => setSearchParams({})} type="button">
            clear set
          </button>
        </div>
      </section>

      <section className="compare-layout">
        <article className="panel panel--quiet">
          <SectionHeader
            eyebrow="Selection"
            title={`Pick 2–4 runs · ${selectedRuns.length}/${MAX_COMPARE}`}
          />
          <div className="selection-list">
            {filteredRuns.map((run) => (
              <button
                key={run.run_id}
                className={`selection-card ${selectedIds.has(run.run_id) ? "is-active" : ""}`}
                disabled={!selectedIds.has(run.run_id) && selectedIds.size >= MAX_COMPARE}
                onClick={() => toggle(run.run_id)}
                type="button"
              >
                <div className="selection-card__header">
                  <strong>{formatRunTitle(run)}</strong>
                  <span className="status-chip is-muted">{formatScore(run.average_score)}</span>
                </div>
                <div className="selection-card__body">
                  <span>{formatRunContext(run)}</span>
                  <span title={run.run_id}>{formatCompactRunId(run.run_id)}</span>
                </div>
                <div className="artifact-strip">
                  {getRunArtifactKinds(run).map((kind) => (
                    <span key={kind}>{kind}</span>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </article>

        <div className="page-stack compare-deck">
          {selectedRuns.length >= 2 ? (
            <>
              <article className="panel">
                <SectionHeader
                  eyebrow="Scores"
                  title="Score spread"
                />
                <CompareBars runs={selectedRuns} />
              </article>

              <article className="panel panel--quiet">
                <SectionHeader
                  eyebrow="References"
                  title="Benchmark anchors"
                />
                <div className="metric-row metric-row--artifact">
                  <MetricCard label="winner private" value={formatScore(OFFICIAL_BENCHMARK.winnerPrivate)} />
                  <MetricCard label="winner public" value={formatScore(OFFICIAL_BENCHMARK.winnerPublic)} />
                  <MetricCard label="RBC private" value={formatScore(OFFICIAL_BENCHMARK.baselinePrivate)} />
                  <MetricCard
                    label="best selected"
                    value={bestSelectedRun ? formatScore(bestSelectedRun.average_score) : "—"}
                    tone="warm"
                    hint={bestSelectedRun ? formatRunTitle(bestSelectedRun) : undefined}
                  />
                </div>
              </article>

              <article className="panel panel--quiet">
                <SectionHeader
                  eyebrow="Delta ledger"
                  title="Selected vs references"
                />
                <div className="table-shell">
                  <table className="runs-table">
                    <thead>
                      <tr>
                        <th>Run</th>
                        <th>Score</th>
                        <th>Vs official RBC private</th>
                        <th>Vs official winner private</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedRuns.map((run) => (
                        <tr key={run.run_id}>
                          <td>
                            <div className="table-cell__stack">
                              <strong>{formatRunTitle(run)}</strong>
                              <span title={run.run_id}>{formatCompactRunId(run.run_id)}</span>
                            </div>
                          </td>
                          <td>{formatScore(run.average_score)}</td>
                          <td>
                            {run.average_score !== null
                              ? formatScore(run.average_score - OFFICIAL_BENCHMARK.baselinePrivate)
                              : "—"}
                          </td>
                          <td>
                            {run.average_score !== null
                              ? formatScore(run.average_score - OFFICIAL_BENCHMARK.winnerPrivate)
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>

              <article className="panel panel--quiet">
                <SectionHeader
                  eyebrow="Challenge matrix"
                  title="Metric matrix"
                />
                <div className="table-shell">
                  <table className="runs-table">
                    <thead>
                      <tr>
                        <th>Run</th>
                        {metricKeys.map((metricKey) => (
                          <th key={metricKey}>{metricKey.replace(/_/g, " ")}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {selectedDetails.map((detail) => (
                        <tr key={detail.summary.run_id}>
                          <td>
                            <div className="table-cell__stack">
                              <strong>{formatRunTitle(detail.summary)}</strong>
                              <span title={detail.summary.run_id}>{formatCompactRunId(detail.summary.run_id)}</span>
                            </div>
                          </td>
                          {metricKeys.map((metricKey) => (
                            <td key={metricKey}>{formatScore(detail.challenge_metrics[metricKey]?.value as number | null)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
            </>
          ) : (
            <article className="panel">
              <div className="empty-block empty-block--wide">
                Select at least two runs from the left to build the comparison deck.
              </div>
            </article>
          )}
        </div>
      </section>
    </div>
  );
}

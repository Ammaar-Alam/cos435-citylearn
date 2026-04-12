import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { ChallengeMetricChart, MetricCard, RunsTable, SectionHeader } from "../components";
import { fetchPlayback, fetchRunDetail, fetchRuns } from "../lib/api";
import {
  artifactUrl,
  formatMetricLabel,
  formatRunContext,
  formatRunTitle,
  formatScore,
  getRunArtifactKinds,
  getRunArtifactPath,
} from "../lib/format";

export function RunsPage() {
  const [query, setQuery] = useState("");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
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

  useEffect(() => {
    if (filteredRuns.length === 0) {
      if (selectedRunId !== null) {
        setSelectedRunId(null);
      }
      return;
    }

    if (!selectedRunId || !filteredRuns.some((run) => run.run_id === selectedRunId)) {
      setSelectedRunId(filteredRuns[0].run_id);
    }
  }, [filteredRuns, selectedRunId]);

  const selectedRun = filteredRuns.find((run) => run.run_id === selectedRunId) ?? null;
  const detailQuery = useQuery({
    queryKey: ["run", selectedRunId],
    queryFn: () => fetchRunDetail(selectedRunId!),
    enabled: Boolean(selectedRunId),
  });
  const playbackQuery = useQuery({
    queryKey: ["playback", selectedRunId],
    queryFn: () => fetchPlayback(selectedRunId!),
    enabled: Boolean(selectedRunId),
  });

  const posterPath = playbackQuery.data?.payload?.media?.poster_path ?? (selectedRun ? getRunArtifactPath(selectedRun, "poster") : null);
  const districtRows = Object.entries(detailQuery.data?.district_kpis ?? {}).slice(0, 6);

  return (
    <div className="page-stack page-stack--runs">
      <section className="page-header">
        <div className="page-header__body">
          <div className="page-header__eyebrow">Runs</div>
          <h1>Run archive</h1>
          <p>Search the ledger, then open the run you need.</p>
        </div>
        <div className="page-header__actions">
          <input
            className="search-input"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="search run, split, or method"
            value={query}
          />
        </div>
      </section>

      <section className="archive-shell">
        <article className="panel panel--quiet">
          <SectionHeader
            eyebrow="Archive"
            title={`${filteredRuns.length} runs`}
          />
          {filteredRuns.length > 0 ? (
            <RunsTable runs={filteredRuns} selectedRunId={selectedRunId} onSelect={setSelectedRunId} />
          ) : (
            <div className="empty-block empty-block--wide">No runs matched the current search.</div>
          )}
        </article>

        <article className="panel">
          {selectedRun ? (
            <>
              <SectionHeader
                eyebrow="Selected run"
                title={formatRunTitle(selectedRun)}
                copy={formatRunContext(selectedRun)}
                action={
                  <div className="inline-actions">
                    <Link className="primary-button" to={`/runs/${selectedRun.run_id}`}>
                      open run
                    </Link>
                    <Link className="ghost-button ghost-button--small" to={`/compare?runIds=${selectedRun.run_id}`}>
                      compare
                    </Link>
                  </div>
                }
              />

              <div className="selected-run-card">
                <div className="selected-run-card__media">
                  {posterPath ? (
                    <img alt="Selected run poster" src={artifactUrl(posterPath) ?? ""} />
                  ) : (
                    <div className="scene-placeholder">No poster captured for this run.</div>
                  )}
                </div>
                <div className="selected-run-card__body">
                  <div className="metric-row metric-row--artifact">
                    <MetricCard label="score" value={formatScore(selectedRun.average_score)} tone="warm" />
                    <MetricCard label="split" value={selectedRun.split} hint={`seed ${selectedRun.seed}`} />
                    <MetricCard label="steps" value={String(selectedRun.step_count)} />
                    <MetricCard
                      label="files"
                      value={String(getRunArtifactKinds(selectedRun).length)}
                      hint={getRunArtifactKinds(selectedRun).join(" • ")}
                    />
                  </div>
                </div>
              </div>

              {detailQuery.data ? (
                <div className="selected-run-grid">
                  <div className="chart-frame">
                    <div className="panel__title">Score anatomy</div>
                    <ChallengeMetricChart detail={detailQuery.data} />
                  </div>
                  <div className="chart-frame">
                    <div className="panel__title">District KPIs</div>
                    <div className="fact-list">
                      {districtRows.map(([key, value]) => (
                        <div key={key} className="fact-list__row">
                          <span>{formatMetricLabel(key)}</span>
                          <strong>{formatScore(value)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <div className="empty-block empty-block--wide">Select a run from the archive to build the briefing panel.</div>
          )}
        </article>
      </section>
    </div>
  );
}

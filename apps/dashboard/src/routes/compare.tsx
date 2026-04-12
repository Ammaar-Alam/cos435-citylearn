import { useQueries, useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

import { CompareBars, RunsTable, SectionHeader } from "../components";
import { fetchRunDetail, fetchRuns } from "../lib/api";
import { formatScore } from "../lib/format";

const officialBenchmark = {
  baselinePrivate: 1.124,
  baselinePublic: 1.085,
  winnerPrivate: 0.565,
  winnerPublic: 0.562,
};

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: fetchRuns });
  const selectedIds = new Set((searchParams.get("runIds") ?? "").split(",").filter(Boolean));
  const runs = runsQuery.data ?? [];
  const selectedRuns = runs.filter((run) => selectedIds.has(run.run_id));
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
    new Set(selectedDetails.flatMap((detail) => Object.keys(detail?.challenge_metrics ?? {}))),
  ).filter((key) => key !== "average_score");
  const bestSelectedScore = selectedRuns.reduce<number | null>((best, run) => {
    if (run.average_score === null) {
      return best;
    }
    if (best === null) {
      return run.average_score;
    }
    return Math.min(best, run.average_score);
  }, null);
  const leaderboardRows = [
    {
      label: "official winner private",
      kind: "reference",
      score: officialBenchmark.winnerPrivate,
    },
    {
      label: "official winner public",
      kind: "reference",
      score: officialBenchmark.winnerPublic,
    },
    {
      label: "official RBC public",
      kind: "reference",
      score: officialBenchmark.baselinePublic,
    },
    {
      label: "official RBC private",
      kind: "reference",
      score: officialBenchmark.baselinePrivate,
    },
    ...selectedRuns
      .filter((run) => run.average_score !== null)
      .map((run) => ({
        label: run.run_id,
        kind: "local",
        score: run.average_score as number,
      })),
  ].sort((left, right) => left.score - right.score);

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
            eyebrow="benchmark anchors"
            title="Private vs public reference values"
            copy="These reference numbers stay visible so local run comparisons keep the real challenge scale in view."
          />
          <div className="metric-row metric-row--artifact">
            <div className="metric-card metric-card--warm">
              <div className="metric-card__label">official RBC private</div>
              <div className="metric-card__value">{formatScore(officialBenchmark.baselinePrivate)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-card__label">official RBC public</div>
              <div className="metric-card__value">{formatScore(officialBenchmark.baselinePublic)}</div>
            </div>
            <div className="metric-card metric-card--mint">
              <div className="metric-card__label">winner private</div>
              <div className="metric-card__value">{formatScore(officialBenchmark.winnerPrivate)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-card__label">winner public</div>
              <div className="metric-card__value">{formatScore(officialBenchmark.winnerPublic)}</div>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="delta ledger"
          title="Selected run deltas"
          copy="This keeps local results grounded against both the best selected run and the official challenge references."
        />
        <div className="table-shell">
          <table className="runs-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Score</th>
                <th>Vs selected best</th>
                <th>Vs official RBC private</th>
                <th>Vs official winner private</th>
              </tr>
            </thead>
            <tbody>
              {selectedRuns.map((run) => (
                <tr key={run.run_id}>
                  <td>{run.run_id}</td>
                  <td>{formatScore(run.average_score)}</td>
                  <td>
                    {run.average_score !== null && bestSelectedScore !== null
                      ? formatScore(run.average_score - bestSelectedScore)
                      : "—"}
                  </td>
                  <td>
                    {run.average_score !== null
                      ? formatScore(run.average_score - officialBenchmark.baselinePrivate)
                      : "—"}
                  </td>
                  <td>
                    {run.average_score !== null
                      ? formatScore(run.average_score - officialBenchmark.winnerPrivate)
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="content-grid">
        <div className="panel">
          <SectionHeader
            eyebrow="leaderboard ladder"
            title="Local runs against the official challenge markers"
            copy="This puts the selected local runs on the same ordered ladder as the public and private reference scores."
          />
          <div className="table-shell">
            <table className="runs-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Entry</th>
                  <th>Type</th>
                  <th>Score</th>
                  <th>Gap vs winner private</th>
                </tr>
              </thead>
              <tbody>
                {leaderboardRows.map((row, index) => (
                  <tr key={`${row.kind}-${row.label}`}>
                    <td>{index + 1}</td>
                    <td>{row.label}</td>
                    <td>{row.kind}</td>
                    <td>{formatScore(row.score)}</td>
                    <td>{formatScore(row.score - officialBenchmark.winnerPrivate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="panel">
          <SectionHeader
            eyebrow="improvement ratios"
            title="How much ground each run closes"
            copy="Positive values mean the run improved on the official RBC private baseline. Smaller remaining gap to the winner is better."
          />
          <div className="table-shell">
            <table className="runs-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Score</th>
                  <th>Improvement vs official RBC private</th>
                  <th>Remaining multiple of winner private</th>
                </tr>
              </thead>
              <tbody>
                {selectedRuns.map((run) => {
                  const score = run.average_score;
                  const improvementPct =
                    score === null
                      ? null
                      : ((officialBenchmark.baselinePrivate - score) / officialBenchmark.baselinePrivate) * 100;
                  const winnerMultiple =
                    score === null ? null : score / officialBenchmark.winnerPrivate;
                  return (
                    <tr key={run.run_id}>
                      <td>{run.run_id}</td>
                      <td>{formatScore(score)}</td>
                      <td>{improvementPct === null ? "—" : `${improvementPct.toFixed(1)}%`}</td>
                      <td>{winnerMultiple === null ? "—" : `${winnerMultiple.toFixed(2)}x`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="challenge matrix"
          title="Weighted metric comparison"
          copy="This is the fast interactive version of the score anatomy view. The paper figures still come from the benchmark JSON/CSV scripts."
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
                  <td>{detail.summary.run_id}</td>
                  {metricKeys.map((metricKey) => (
                    <td key={metricKey}>{formatScore(detail.challenge_metrics[metricKey]?.value as number | null)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="selected runs"
          title="Run ledger"
          copy="Keep the comparison set small enough that the deltas stay interpretable."
        />
        <RunsTable runs={selectedRuns} />
      </section>
    </div>
  );
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  cancelJob,
  createJob,
  fetchEnvironment,
  fetchJobLogs,
  fetchJobs,
  fetchRunners,
  fetchRuns,
} from "../lib/api";
import { OFFICIAL_BENCHMARK } from "../lib/benchmark";
import {
  artifactUrl,
  formatRunContext,
  formatRunTitle,
  formatScore,
  formatShortHash,
  getRunArtifactKinds,
  getRunArtifactPath,
} from "../lib/format";
import { useInterval } from "../lib/useInterval";
import { JobsRail, MetricCard, RunnerGrid, RunsTable, SectionHeader } from "../components";

export function OverviewPage() {
  const queryClient = useQueryClient();
  const [selectedRunnerId, setSelectedRunnerId] = useState("rbc_builtin");
  const [seed, setSeed] = useState(0);
  const [traceLimit, setTraceLimit] = useState(96);
  const [captureFrames, setCaptureFrames] = useState(true);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const envQuery = useQuery({ queryKey: ["env"], queryFn: fetchEnvironment });
  const runnersQuery = useQuery({ queryKey: ["runners"], queryFn: fetchRunners });
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: fetchRuns });
  const jobsQuery = useQuery({ queryKey: ["jobs"], queryFn: fetchJobs });

  const effectiveSelectedJobId = useMemo(() => {
    if (selectedJobId) {
      return selectedJobId;
    }

    const jobs = jobsQuery.data ?? [];
    return jobs.find((job) => job.status === "running" || job.status === "failed")?.job_id ?? jobs[0]?.job_id ?? null;
  }, [jobsQuery.data, selectedJobId]);

  const jobLogsQuery = useQuery({
    queryKey: ["job-logs", effectiveSelectedJobId],
    queryFn: () => fetchJobLogs(effectiveSelectedJobId!),
    enabled: Boolean(effectiveSelectedJobId),
  });

  useInterval(() => {
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    queryClient.invalidateQueries({ queryKey: ["runs"] });
    if (effectiveSelectedJobId) {
      queryClient.invalidateQueries({ queryKey: ["job-logs", effectiveSelectedJobId] });
    }
  }, 3000);

  const launchMutation = useMutation({
    mutationFn: createJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const latestRun = runsQuery.data?.[0] ?? null;
  const bestLocalRun = useMemo(() => {
    const runs = runsQuery.data ?? [];

    return runs.reduce<typeof runs[number] | null>((best, run) => {
      if (run.average_score === null) {
        return best;
      }
      if (best === null || best.average_score === null) {
        return run;
      }
      return run.average_score < best.average_score ? run : best;
    }, null);
  }, [runsQuery.data]);

  const selectedJob = (jobsQuery.data ?? []).find((job) => job.job_id === effectiveSelectedJobId) ?? null;
  const launchableRunner = runnersQuery.data?.find((runner) => runner.runner_id === selectedRunnerId) ?? null;
  const availableRunners = (runnersQuery.data ?? []).filter((runner) => runner.launchable);
  const plannedRunners = (runnersQuery.data ?? []).filter((runner) => !runner.launchable);
  const latestRunPoster = latestRun ? artifactUrl(getRunArtifactPath(latestRun, "poster")) : null;
  const latestRunArtifacts = latestRun ? getRunArtifactKinds(latestRun) : [];
  const bestLocalGap = bestLocalRun?.average_score !== null && bestLocalRun?.average_score !== undefined
    ? bestLocalRun.average_score - OFFICIAL_BENCHMARK.winnerPrivate
    : null;
  const latestRunGap = latestRun?.average_score !== null && latestRun?.average_score !== undefined
    ? latestRun.average_score - OFFICIAL_BENCHMARK.baselinePrivate
    : null;

  const environmentPackages = (envQuery.data?.packages ?? {}) as Record<string, string>;
  const environmentRows = [
    { label: "python", value: String(envQuery.data?.python ?? "—") },
    { label: "CityLearn", value: environmentPackages.CityLearn ?? "—" },
    { label: "commit", value: formatShortHash(String(envQuery.data?.git_commit ?? "")) },
  ];

  return (
    <div className="page-stack page-stack--overview">
      <section className="page-header">
        <div className="page-header__body">
          <div className="page-header__eyebrow">Overview</div>
          <h1>Benchmark desk</h1>
          <p>Latest run, launch control, and queue state.</p>
        </div>
        <div className="page-header__actions">
          <Link className="primary-button" to={latestRun ? `/runs/${latestRun.run_id}` : "/runs"}>
            latest run
          </Link>
          <Link className="ghost-button" to="/monitor">
            live view
          </Link>
        </div>
      </section>

      <section className="overview-hero">
        <article className="panel latest-run-card">
          <div className="latest-run-card__media">
            {latestRunPoster ? (
              <img alt="Latest run poster" src={latestRunPoster} />
            ) : (
              <div className="scene-placeholder">No poster captured yet.</div>
            )}
          </div>
          <div className="latest-run-card__body">
            <div className="panel__title">Latest run</div>
            <h2>{latestRun ? formatRunTitle(latestRun) : "No completed runs yet"}</h2>
            <p>{latestRun ? formatRunContext(latestRun) : "Launch the first benchmark run."}</p>
            <div className="metric-row">
              <MetricCard
                label="score"
                value={latestRun ? formatScore(latestRun.average_score) : "—"}
                tone="warm"
                hint="lower is better"
              />
              <MetricCard
                label="vs RBC private"
                value={latestRunGap === null ? "—" : formatScore(latestRunGap)}
                hint="local minus baseline"
              />
              <MetricCard
                label="files"
                value={latestRunArtifacts.length ? String(latestRunArtifacts.length) : "0"}
                hint={latestRunArtifacts.length ? latestRunArtifacts.join(" • ") : "pending"}
              />
            </div>
            {latestRun ? (
              <div className="inline-actions">
                <Link className="ghost-button" to={`/runs/${latestRun.run_id}`}>
                  open run
                </Link>
                <Link className="ghost-button" to={`/compare?runIds=${latestRun.run_id}`}>
                  compare
                </Link>
              </div>
            ) : null}
          </div>
        </article>

        <div className="overview-rail">
          <article className="panel panel--quiet">
            <SectionHeader eyebrow="Benchmark" title="Reference scores" />
            <div className="reference-list">
              <div className="reference-row">
                <span>winner private</span>
                <strong>{formatScore(OFFICIAL_BENCHMARK.winnerPrivate)}</strong>
              </div>
              <div className="reference-row">
                <span>winner public</span>
                <strong>{formatScore(OFFICIAL_BENCHMARK.winnerPublic)}</strong>
              </div>
              <div className="reference-row">
                <span>RBC private</span>
                <strong>{formatScore(OFFICIAL_BENCHMARK.baselinePrivate)}</strong>
              </div>
              <div className="reference-row">
                <span>best local</span>
                <strong>{bestLocalRun ? formatScore(bestLocalRun.average_score) : "—"}</strong>
              </div>
              <div className="reference-row">
                <span>best local gap</span>
                <strong>{bestLocalGap === null ? "—" : formatScore(bestLocalGap)}</strong>
              </div>
            </div>
          </article>

          <article className="panel panel--quiet">
            <SectionHeader eyebrow="System" title="Bench lock" />
            <div className="fact-list">
              {environmentRows.map((row) => (
                <div key={row.label} className="fact-list__row">
                  <span>{row.label}</span>
                  <strong>{row.value}</strong>
                </div>
              ))}
            </div>
          </article>
        </div>
      </section>

      <section className="split-grid">
        <article className="panel">
          <SectionHeader
            eyebrow="Launch"
            title="Run control"
          />
          {availableRunners.length > 0 ? (
            <RunnerGrid
              runners={availableRunners}
              selectedRunnerId={selectedRunnerId}
              onSelect={setSelectedRunnerId}
            />
          ) : (
            <div className="empty-block">No launchable runners are registered.</div>
          )}

          <details className="detail-block" open>
            <summary>Run settings</summary>
            <div className="launch-form">
              <label>
                seed
                <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
              </label>
              <label>
                trace limit
                <input
                  type="number"
                  value={traceLimit}
                  onChange={(event) => setTraceLimit(Number(event.target.value))}
                />
              </label>
              <label className="checkbox-row">
                <input
                  checked={captureFrames}
                  onChange={(event) => setCaptureFrames(event.target.checked)}
                  type="checkbox"
                />
                capture render frames
              </label>
              <button
                className="primary-button"
                disabled={!launchableRunner?.launchable || launchMutation.isPending}
                onClick={() =>
                  launchMutation.mutate(
                    {
                      runner_id: selectedRunnerId,
                      seed,
                      trace_limit: traceLimit,
                      capture_render_frames: captureFrames,
                    },
                    {
                      onSuccess: (job) => {
                        setSelectedJobId(job.job_id);
                      },
                    },
                  )
                }
                type="button"
              >
                {launchMutation.isPending ? "launching…" : "launch benchmark"}
              </button>
            </div>
          </details>

          {plannedRunners.length > 0 ? (
            <div className="planned-runners">
              <div className="panel__title">Queued next</div>
              <div className="artifact-strip">
                {plannedRunners.map((runner) => (
                  <span key={runner.runner_id}>{runner.label}</span>
                ))}
              </div>
            </div>
          ) : null}
        </article>

        <article className="panel panel--quiet">
          <SectionHeader
            eyebrow="Queue"
            title="Active jobs"
            action={
              <Link className="ghost-button ghost-button--small" to="/monitor">
                live
              </Link>
            }
          />
          <JobsRail
            jobs={jobsQuery.data ?? []}
            selectedJobId={effectiveSelectedJobId}
            onSelect={setSelectedJobId}
            onCancel={(jobId) => cancelMutation.mutate(jobId)}
            limit={4}
          />
          <details className="detail-block">
            <summary>Latest log tail</summary>
            <pre className="job-log-output">
              {selectedJob ? jobLogsQuery.data?.logs?.trim() || "Waiting for log output." : "Select a job to inspect its log."}
            </pre>
          </details>
        </article>
      </section>

      <section className="content-grid">
        <article className="panel panel--quiet">
          <SectionHeader
            eyebrow="Archive"
            title="Recent runs"
          />
          <RunsTable runs={runsQuery.data ?? []} limit={8} />
        </article>

        <article className="panel panel--quiet">
          <SectionHeader
            eyebrow="Best local"
            title={bestLocalRun ? formatRunTitle(bestLocalRun) : "No local benchmark yet"}
            copy={bestLocalRun ? formatRunContext(bestLocalRun) : "Run the benchmark to establish a local reference."}
          />
          {bestLocalRun ? (
            <>
              <div className="metric-row metric-row--artifact">
                <MetricCard label="score" value={formatScore(bestLocalRun.average_score)} tone="mint" />
                <MetricCard
                  label="gap vs winner"
                  value={bestLocalGap === null ? "—" : formatScore(bestLocalGap)}
                  hint="local minus winner"
                />
                <MetricCard
                  label="files"
                  value={String(getRunArtifactKinds(bestLocalRun).length)}
                  hint={getRunArtifactKinds(bestLocalRun).join(" • ")}
                />
                <MetricCard label="split" value={bestLocalRun.split} hint={`seed ${bestLocalRun.seed}`} />
              </div>
              <div className="inline-actions">
                <Link className="ghost-button" to={`/runs/${bestLocalRun.run_id}`}>
                  open run
                </Link>
                <Link className="ghost-button" to={`/compare?runIds=${bestLocalRun.run_id}`}>
                  compare
                </Link>
              </div>
            </>
          ) : (
            <div className="empty-block empty-block--wide">The first completed run will show up here automatically.</div>
          )}
        </article>
      </section>
    </div>
  );
}

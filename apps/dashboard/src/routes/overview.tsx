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
import { formatScore } from "../lib/format";
import { useInterval } from "../lib/useInterval";
import { JobsRail, MetricCard, RunnerGrid, RunsTable, SectionHeader } from "../components";

const officialBenchmark = {
  baselinePrivate: 1.124,
  baselinePublic: 1.085,
  winnerPrivate: 0.565,
  winnerPublic: 0.562,
};

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
      if (effectiveSelectedJobId) {
        queryClient.invalidateQueries({ queryKey: ["job-logs", effectiveSelectedJobId] });
      }
    },
  });

  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      if (effectiveSelectedJobId) {
        queryClient.invalidateQueries({ queryKey: ["job-logs", effectiveSelectedJobId] });
      }
    },
  });

  const latestRun = runsQuery.data?.[0];
  const launchableRunner = runnersQuery.data?.find((runner) => runner.runner_id === selectedRunnerId);
  const benchmarkGap = useMemo(
    () => officialBenchmark.baselinePrivate - officialBenchmark.winnerPrivate,
    [],
  );

  return (
    <div className="page-stack">
      <section className="hero">
        <div className="hero__copy">
          <div className="hero__eyebrow">local benchmark dashboard</div>
          <h1>
            Launch the benchmark, watch completed evaluations, and keep the paper story tied to the same artifacts.
          </h1>
          <p>
            The backend remains the source of truth. Every completed eval exports run metrics, playback payloads, and official-style simulation files by default.
          </p>
          <div className="hero__actions">
            <Link className="primary-button" to={latestRun ? `/runs/${latestRun.run_id}` : "/runs"}>
              inspect latest run
            </Link>
            <Link className="ghost-button" to="/compare">
              compare runs
            </Link>
          </div>
        </div>
        <div className="hero__stats">
          <MetricCard label="official RBC private" value={formatScore(officialBenchmark.baselinePrivate)} tone="warm" />
          <MetricCard label="winner private" value={formatScore(officialBenchmark.winnerPrivate)} tone="mint" />
          <MetricCard label="benchmark gap" value={formatScore(benchmarkGap)} hint="private baseline minus winner" />
          <MetricCard
            label="latest local run"
            value={latestRun ? formatScore(latestRun.average_score) : "—"}
            hint={latestRun ? latestRun.run_id : "run RBC from the launch panel"}
          />
        </div>
      </section>

      <section className="split-grid">
        <div className="panel panel--feature">
          <SectionHeader
            eyebrow="launch"
            title="Preset runner control"
            copy="Current launch support is honest: the dashboard can run the same built-in RBC benchmark path that exists in the repo today."
          />
          <RunnerGrid
            runners={runnersQuery.data ?? []}
            selectedRunnerId={selectedRunnerId}
            onSelect={setSelectedRunnerId}
          />

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
              capture literal render frames
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
              {launchMutation.isPending ? "launching…" : "launch benchmark job"}
            </button>
          </div>
        </div>

        <div className="panel">
          <SectionHeader
            eyebrow="runtime"
            title="Environment status"
            copy="This reads the same Python environment lock the repo already uses for benchmark verification."
          />
          <div className="env-list">
            {Object.entries(envQuery.data ?? {}).map(([key, value]) => (
              <div key={key} className="env-list__row">
                <span>{key}</span>
                <strong>{typeof value === "object" ? JSON.stringify(value) : String(value)}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="content-grid">
        <div className="panel panel--table">
          <SectionHeader
            eyebrow="runs"
            title="Recent benchmark runs"
            copy="These are discovered from the tracked run artifacts on disk. The CSV row is derivative; the dashboard reads the canonical JSON."
          />
          <RunsTable runs={runsQuery.data ?? []} />
        </div>

        <div className="panel">
          <SectionHeader
            eyebrow="jobs"
            title="Active and recent jobs"
            copy="The dashboard launches a subprocess worker, tails the same process log on disk, and then discovers the finished run from the benchmark artifacts."
          />
          <JobsRail
            jobs={jobsQuery.data ?? []}
            selectedJobId={effectiveSelectedJobId}
            onSelect={setSelectedJobId}
            onCancel={(jobId) => cancelMutation.mutate(jobId)}
          />
          <div className="job-log-panel">
            <div className="panel__title">selected job log</div>
            <pre className="job-log-output">
              {jobLogsQuery.data?.logs?.trim() || "select a job to inspect its latest log output"}
            </pre>
          </div>
        </div>
      </section>
    </div>
  );
}

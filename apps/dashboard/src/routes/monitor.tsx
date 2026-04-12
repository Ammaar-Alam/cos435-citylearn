import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  cancelJob,
  fetchJobArtifacts,
  fetchJobEvents,
  fetchJobLogs,
  fetchJobPreview,
  fetchJobs,
  fetchJobState,
} from "../lib/api";
import {
  artifactUrl,
  basename,
  formatRelativeTime,
  formatScore,
} from "../lib/format";
import { useInterval } from "../lib/useInterval";
import { JobsRail, MetricCard, PlaybackScene, SectionHeader, TimeseriesPanel, TraceTable } from "../components";
import type { JobEvent } from "../types";

function formatProgress(current: number | null, total: number | null): string {
  if (current === null || total === null || total <= 0) {
    return "—";
  }

  return `${Math.min(100, (current / total) * 100).toFixed(0)}%`;
}

function summarizeEvent(event: JobEvent): { title: string; note: string } {
  if (event.event_type === "job_submitted") {
    return { title: "Job submitted", note: event.payload.runner_id ?? "runner queued" };
  }
  if (event.event_type === "process_spawned") {
    return { title: "Worker spawned", note: `pid ${String(event.payload.pid ?? "—")}` };
  }
  if (event.event_type === "job_started") {
    return {
      title: "Rollout started",
      note: `${String(event.payload.progress_label ?? "benchmark rollout")} · ${String(event.payload.progress_total ?? "—")} steps`,
    };
  }
  if (event.event_type === "progress") {
    return {
      title: "Progress checkpoint",
      note: `${String(event.payload.progress_current ?? "—")}/${String(event.payload.progress_total ?? "—")} · ${String(event.payload.progress_label ?? "rollout")}`,
    };
  }
  if (event.event_type === "artifact_written") {
    return {
      title: String(event.payload.label ?? "Artifact written"),
      note: basename(String(event.payload.path ?? "")),
    };
  }
  if (event.event_type === "job_finished") {
    return {
      title: "Run completed",
      note: event.payload.average_score !== null && event.payload.average_score !== undefined
        ? `score ${formatScore(Number(event.payload.average_score))}`
        : "result written",
    };
  }

  return {
    title: event.event_type.replace(/_/g, " "),
    note: Object.entries(event.payload).slice(0, 2).map(([key, value]) => `${key}: ${String(value)}`).join(" · "),
  };
}

function buildMilestones(events: JobEvent[]): JobEvent[] {
  const milestones: JobEvent[] = [];
  let lastBucket = -1;

  for (const event of events) {
    if (event.event_type !== "progress") {
      milestones.push(event);
      continue;
    }

    const current = Number(event.payload.progress_current ?? 0);
    const total = Number(event.payload.progress_total ?? 0);
    const bucket = total > 0 ? Math.floor((current / total) * 4) : -1;
    const isLast = total > 0 && current >= total;

    if (bucket > lastBucket || isLast) {
      milestones.push(event);
      lastBucket = Math.max(lastBucket, bucket);
    }
  }

  return milestones.slice(-8);
}

export function MonitorPage() {
  const queryClient = useQueryClient();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedBuildingIndex, setSelectedBuildingIndex] = useState(0);
  const [autoFollow, setAutoFollow] = useState(true);
  const [stepIndex, setStepIndex] = useState(0);

  const jobsQuery = useQuery({ queryKey: ["jobs"], queryFn: fetchJobs });
  const effectiveJobId = useMemo(() => {
    if (selectedJobId) {
      return selectedJobId;
    }

    const jobs = jobsQuery.data ?? [];
    return (
      jobs.find((job) => job.status === "running" && job.latest_preview_path)?.job_id ??
      jobs.find((job) => job.status === "running")?.job_id ??
      jobs.find((job) => job.status === "queued")?.job_id ??
      jobs.find((job) => job.latest_preview_path)?.job_id ??
      jobs[0]?.job_id ??
      null
    );
  }, [jobsQuery.data, selectedJobId]);

  const selectedJob = jobsQuery.data?.find((job) => job.job_id === effectiveJobId) ?? null;

  const stateQuery = useQuery({
    queryKey: ["job-state", effectiveJobId],
    queryFn: () => fetchJobState(effectiveJobId!),
    enabled: Boolean(effectiveJobId),
  });
  const previewQuery = useQuery({
    queryKey: ["job-preview", effectiveJobId],
    queryFn: () => fetchJobPreview(effectiveJobId!),
    enabled: Boolean(stateQuery.data?.latest_preview_path ?? selectedJob?.latest_preview_path),
    retry: false,
  });
  const eventsQuery = useQuery({
    queryKey: ["job-events", effectiveJobId],
    queryFn: () => fetchJobEvents(effectiveJobId!),
    enabled: Boolean(effectiveJobId),
  });
  const logsQuery = useQuery({
    queryKey: ["job-logs", effectiveJobId],
    queryFn: () => fetchJobLogs(effectiveJobId!),
    enabled: Boolean(effectiveJobId),
  });
  const artifactsQuery = useQuery({
    queryKey: ["job-artifacts", effectiveJobId],
    queryFn: () => fetchJobArtifacts(effectiveJobId!),
    enabled: Boolean(effectiveJobId),
  });

  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      if (effectiveJobId) {
        queryClient.invalidateQueries({ queryKey: ["job-state", effectiveJobId] });
      }
    },
  });

  useInterval(() => {
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    if (effectiveJobId) {
      queryClient.invalidateQueries({ queryKey: ["job-state", effectiveJobId] });
      queryClient.invalidateQueries({ queryKey: ["job-preview", effectiveJobId] });
      queryClient.invalidateQueries({ queryKey: ["job-events", effectiveJobId] });
      queryClient.invalidateQueries({ queryKey: ["job-logs", effectiveJobId] });
    }
  }, 1800);

  const preview = previewQuery.data;
  const state = stateQuery.data;

  useEffect(() => {
    if (!preview) {
      return;
    }

    const latestIndex = Math.max(0, preview.stored_steps - 1);
    if (autoFollow) {
      setStepIndex(latestIndex);
      return;
    }

    setStepIndex((current) => Math.min(current, latestIndex));
  }, [autoFollow, preview?.stored_steps, preview]);

  useEffect(() => {
    const buildingCount = preview?.payload?.buildings?.length ?? preview?.building_names?.length ?? 0;
    if (selectedBuildingIndex >= buildingCount) {
      setSelectedBuildingIndex(0);
    }
  }, [preview?.building_names?.length, preview?.payload?.buildings?.length, selectedBuildingIndex]);

  const eventRows = eventsQuery.data ?? [];
  const milestones = useMemo(() => buildMilestones(eventRows), [eventRows]);
  const media = preview?.payload?.media ?? {};
  const previewArtifacts = [
    { kind: "simulation_export", label: "simulation export", path: preview?.payload?.ui_export?.simulation_dir ?? null },
    { kind: "poster", label: "poster frame", path: media.poster_path ?? null },
    { kind: "gif", label: "gif playback", path: media.gif_path ?? null },
  ].filter((item): item is { kind: string; label: string; path: string } => Boolean(item.path));
  const persistedArtifacts = artifactsQuery.data ?? [];
  const artifactRows = [...persistedArtifacts];
  for (const item of previewArtifacts) {
    if (!artifactRows.some((existing) => existing.path === item.path)) {
      artifactRows.push(item);
    }
  }

  return (
    <div className="page-stack page-stack--monitor">
      <section className="page-header">
        <div className="page-header__body">
          <div className="page-header__eyebrow">Live</div>
          <h1>Live monitor</h1>
          <p>Selected job, preview, trace.</p>
        </div>
        <div className="page-header__actions">
          {selectedJob?.run_id ? (
            <Link className="primary-button" to={`/runs/${selectedJob.run_id}`}>
              open run
            </Link>
          ) : null}
        </div>
      </section>

      <section className="monitor-shell">
        <article className="panel">
          <SectionHeader
            eyebrow="Preview"
            title={selectedJob ? selectedJob.runner_id.replace(/_/g, " ") : "Waiting for preview"}
            copy={selectedJob?.run_id ?? "Pick a job to begin monitoring."}
          />
          {preview ? (
            <>
              <PlaybackScene playback={preview} stepIndex={Math.min(stepIndex, Math.max(preview.stored_steps - 1, 0))} />
              <div className="playback-controls">
                <label className="checkbox-row">
                  <input checked={autoFollow} onChange={(event) => setAutoFollow(event.target.checked)} type="checkbox" />
                  follow live
                </label>
                <input
                  className="playback-slider"
                  max={Math.max((preview?.stored_steps ?? 1) - 1, 0)}
                  min={0}
                  onChange={(event) => {
                    setAutoFollow(false);
                    setStepIndex(Number(event.target.value));
                  }}
                  type="range"
                  value={Math.min(stepIndex, Math.max((preview?.stored_steps ?? 1) - 1, 0))}
                />
                <span>step {preview ? Math.min(stepIndex, Math.max(preview.stored_steps - 1, 0)) : 0}</span>
              </div>
            </>
          ) : (
            <div className="empty-block empty-block--wide">No live preview yet. Launch a run or wait for the first preview heartbeat.</div>
          )}
        </article>

        <div className="monitor-rail">
          <article className="panel panel--quiet">
            <SectionHeader eyebrow="Status" title="Job state" />
            <div className="metric-row metric-row--artifact">
              <MetricCard label="runner" value={selectedJob?.runner_id ?? "—"} />
              <MetricCard label="status" value={selectedJob?.status ?? "idle"} tone="mint" />
              <MetricCard label="phase" value={state?.phase ?? selectedJob?.phase ?? "—"} />
              <MetricCard
                label="progress"
                value={formatProgress(state?.progress_current ?? null, state?.progress_total ?? null)}
                hint={state?.progress_label ?? "no live label"}
                tone="warm"
              />
            </div>
          </article>

          <article className="panel panel--quiet">
            <SectionHeader eyebrow="Queue" title="Recent jobs" />
            <JobsRail
              jobs={jobsQuery.data ?? []}
              selectedJobId={effectiveJobId}
              onSelect={setSelectedJobId}
              onCancel={(jobId) => cancelMutation.mutate(jobId)}
              limit={4}
            />
          </article>

          <article className="panel panel--quiet">
            <SectionHeader eyebrow="Outputs" title="Captured files" />
            <div className="artifact-list">
              {artifactRows.length === 0 ? (
                <div className="empty-block">No artifacts written yet.</div>
              ) : (
                artifactRows.map((artifact) => (
                  <a key={artifact.path} className="artifact-row" href={artifactUrl(artifact.path) ?? "#"} target="_blank" rel="noreferrer">
                    <strong>{artifact.label}</strong>
                    <span>{basename(artifact.path)}</span>
                  </a>
                ))
              )}
            </div>
          </article>
        </div>
      </section>

      {preview ? (
        <TimeseriesPanel
          playback={preview}
          stepIndex={Math.min(stepIndex, Math.max(preview.stored_steps - 1, 0))}
          selectedBuildingIndex={selectedBuildingIndex}
          onSelectBuilding={setSelectedBuildingIndex}
        />
      ) : null}

      <section className="content-grid">
        <article className="panel panel--quiet">
          <SectionHeader eyebrow="Milestones" title="Event log" />
          <div className="timeline-list">
            {milestones.length === 0 ? (
              <div className="empty-block">No milestones yet.</div>
            ) : (
              milestones.map((event) => {
                const summary = summarizeEvent(event);

                return (
                  <div key={event.seq} className="timeline-row">
                    <div className="timeline-row__meta">
                      <strong>{summary.title}</strong>
                      <span>{formatRelativeTime(event.created_at)}</span>
                    </div>
                    <span>{summary.note}</span>
                  </div>
                );
              })
            )}
          </div>
        </article>

        <article className="panel panel--quiet">
          <SectionHeader eyebrow="Logs" title="Worker log" />
          <pre className="job-log-output">
            {selectedJob ? logsQuery.data?.logs?.trim() || "Waiting for log output." : "Select a job to inspect its log."}
          </pre>
        </article>
      </section>

      <article className="panel panel--quiet">
        <SectionHeader eyebrow="Trace" title="Current slice" />
        {preview ? (
          <TraceTable
            frames={preview.trace_frames}
            stepIndex={Math.min(stepIndex, Math.max(preview.stored_steps - 1, 0))}
            windowSize={12}
          />
        ) : (
          <div className="empty-block">No trace frames yet.</div>
        )}
      </article>
    </div>
  );
}

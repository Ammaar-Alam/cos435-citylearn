import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  fetchJobArtifacts,
  cancelJob,
  fetchJobEvents,
  fetchJobLogs,
  fetchJobPreview,
  fetchJobs,
  fetchJobState,
} from "../lib/api";
import { artifactUrl, formatRelativeTime, formatScore } from "../lib/format";
import { useInterval } from "../lib/useInterval";
import { JobsRail, MetricCard, PlaybackScene, SectionHeader, TimeseriesPanel, TraceTable } from "../components";

function formatProgress(current: number | null, total: number | null): string {
  if (current === null || total === null || total <= 0) {
    return "—";
  }

  return `${Math.min(100, (current / total) * 100).toFixed(0)}%`;
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
    <div className="page-stack">
      <section className="panel panel--feature">
        <SectionHeader
          eyebrow="live monitor"
          title="Watch the benchmark worker while it is still writing artifacts"
          copy="The monitor reads job state, live preview payloads, events, and process logs from disk. Later checkpoint-eval training can plug into the same surface without changing the benchmark artifact contract."
        />
        <div className="metric-row">
          <MetricCard label="selected job" value={selectedJob?.runner_id ?? "—"} />
          <MetricCard label="status" value={selectedJob?.status ?? "idle"} tone="mint" />
          <MetricCard label="phase" value={state?.phase ?? selectedJob?.phase ?? "—"} />
          <MetricCard
            label="progress"
            value={formatProgress(state?.progress_current ?? null, state?.progress_total ?? null)}
            hint={state?.progress_label ?? "waiting for live state"}
            tone="warm"
          />
          <MetricCard
            label="run id"
            value={selectedJob?.run_id ?? state?.latest_run_id ?? "—"}
            hint={selectedJob?.submitted_at ? `submitted ${formatRelativeTime(selectedJob.submitted_at)}` : undefined}
          />
          <MetricCard
            label="live trace"
            value={preview ? `${preview.stored_steps}/${preview.total_steps}` : "—"}
            hint={preview ? (preview.truncated ? "preview is still growing" : "full episode captured") : "no preview yet"}
          />
        </div>
      </section>

      <section className="content-grid">
        <div className="panel">
          <SectionHeader
            eyebrow="jobs"
            title="Current queue"
            copy="Pick the active job you want to follow. The monitor defaults to the newest running job."
          />
          <JobsRail
            jobs={jobsQuery.data ?? []}
            selectedJobId={effectiveJobId}
            onSelect={setSelectedJobId}
            onCancel={(jobId) => cancelMutation.mutate(jobId)}
          />
        </div>
        <div className="panel">
          <SectionHeader
            eyebrow="events"
            title="Job timeline"
            copy="Each progress heartbeat and artifact write is recorded so the live view can stay anchored to real worker state."
          />
          <div className="timeline-list">
            {eventRows.length === 0 ? (
              <div className="empty-block">No events yet for this job.</div>
            ) : (
              eventRows
                .slice()
                .reverse()
                .map((event) => (
                  <div key={event.seq} className="timeline-row">
                    <div className="timeline-row__meta">
                      <strong>{event.event_type.replace(/_/g, " ")}</strong>
                      <span>{formatRelativeTime(event.created_at)}</span>
                    </div>
                    <code>{JSON.stringify(event.payload)}</code>
                  </div>
                ))
            )}
          </div>
        </div>
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="preview"
          title="Latest live preview"
          copy="This preview is written by the worker while the rollout progresses. It uses the same playback schema as completed runs, just with a shorter stored trace."
          action={
            <div className="playback-controls">
              <label className="checkbox-row">
                <input checked={autoFollow} onChange={(event) => setAutoFollow(event.target.checked)} type="checkbox" />
                follow latest step
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
          }
        />
        {preview ? (
          <PlaybackScene playback={preview} stepIndex={Math.min(stepIndex, Math.max(preview.stored_steps - 1, 0))} />
        ) : (
          <div className="empty-block empty-block--wide">No live preview yet. Launch an eval job or wait for the first preview heartbeat.</div>
        )}
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
        <div className="panel">
          <SectionHeader
            eyebrow="trace"
            title="Current action slice"
            copy="The monitor table is intentionally local to the selected job, so you can watch the action and reward history accumulate without opening the full run detail page."
          />
          {preview ? (
            <TraceTable frames={preview.trace_frames} stepIndex={Math.min(stepIndex, Math.max(preview.stored_steps - 1, 0))} />
          ) : (
            <div className="empty-block">No trace frames yet.</div>
          )}
        </div>
        <div className="panel">
          <SectionHeader
            eyebrow="artifacts"
            title="Live outputs"
            copy="These are the same files the run detail page will read after the job finishes."
          />
          <div className="artifact-list">
            {artifactRows.length === 0 ? (
              <div className="empty-block">No preview artifacts written yet.</div>
            ) : (
              artifactRows.map((item) => (
                <a
                  key={item.label}
                  className="artifact-row"
                  href={artifactUrl(item.path) ?? undefined}
                  rel="noreferrer"
                  target="_blank"
                >
                  <strong>{item.label}</strong>
                  <span>{item.path}</span>
                </a>
              ))
            )}
          </div>
          {selectedJob?.run_id ? (
            <Link className="ghost-button" to={`/runs/${selectedJob.run_id}`}>
              open completed run
            </Link>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="logs"
          title="Worker log tail"
          copy="This is the direct subprocess output. If the job fails, this is where the first useful signal usually shows up."
        />
        <pre className="job-log-output">
          {logsQuery.data?.logs?.trim() || "No logs yet for the selected job."}
        </pre>
      </section>
    </div>
  );
}

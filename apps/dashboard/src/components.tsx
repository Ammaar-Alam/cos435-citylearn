import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { JobSummary, PlaybackPayload, RunDetail, RunSummary, RunnerSummary } from "./types";
import {
  artifactUrl,
  formatCompactRunId,
  formatDatasetName,
  formatMetricLabel,
  formatRelativeTime,
  formatRunContext,
  formatRunTitle,
  formatScore,
  formatStatusLabel,
  getRunArtifactKinds,
} from "./lib/format";

export function SectionHeader({
  eyebrow,
  title,
  copy,
  action,
}: {
  eyebrow: string;
  title: string;
  copy?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-header">
      <div className="section-header__body">
        <div className="section-header__eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        {copy ? <p>{copy}</p> : null}
      </div>
      {action ? <div className="section-header__action">{action}</div> : null}
    </div>
  );
}

export function MetricCard({
  label,
  value,
  tone = "neutral",
  hint,
}: {
  label: string;
  value: string;
  tone?: "neutral" | "warm" | "mint";
  hint?: string;
}) {
  return (
    <div className={`metric-card metric-card--${tone}`}>
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">{value}</div>
      {hint ? <div className="metric-card__hint">{hint}</div> : null}
    </div>
  );
}

export function RunnerGrid({
  runners,
  selectedRunnerId,
  onSelect,
}: {
  runners: RunnerSummary[];
  selectedRunnerId: string;
  onSelect: (runnerId: string) => void;
}) {
  return (
    <div className="runner-grid">
      {runners.map((runner) => (
        <button
          key={runner.runner_id}
          className={`runner-tile ${selectedRunnerId === runner.runner_id ? "is-active" : ""}`}
          onClick={() => onSelect(runner.runner_id)}
          type="button"
        >
          <div className="runner-tile__topline">
            <span>{runner.algorithm.toUpperCase()}</span>
            <span className={`status-chip ${runner.launchable ? "is-running" : "is-muted"}`}>
              {runner.launchable ? "available" : "planned"}
            </span>
          </div>
          <h3>{runner.label}</h3>
          <p>{runner.description}</p>
        </button>
      ))}
    </div>
  );
}

export function RunsTable({
  runs,
  selectedRunId,
  onSelect,
  limit,
}: {
  runs: RunSummary[];
  selectedRunId?: string | null;
  onSelect?: (runId: string) => void;
  limit?: number;
}) {
  const visibleRuns = limit ? runs.slice(0, limit) : runs;

  return (
    <div className="table-shell">
      <table className="runs-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Setup</th>
            <th>Score</th>
            <th>Files</th>
          </tr>
        </thead>
        <tbody>
          {visibleRuns.map((run) => {
            const artifactKinds = getRunArtifactKinds(run);

            return (
              <tr key={run.run_id} className={selectedRunId === run.run_id ? "is-selected" : undefined}>
                <td>
                  {onSelect ? (
                    <button className="table-link" onClick={() => onSelect(run.run_id)} type="button">
                      <span className="table-link__title">{formatRunTitle(run)}</span>
                      <span className="table-link__meta" title={run.run_id}>{formatCompactRunId(run.run_id)}</span>
                    </button>
                  ) : (
                    <Link className="table-link" to={`/runs/${run.run_id}`}>
                      <span className="table-link__title">{formatRunTitle(run)}</span>
                      <span className="table-link__meta" title={run.run_id}>{formatCompactRunId(run.run_id)}</span>
                    </Link>
                  )}
                </td>
                <td>
                  <div className="table-cell__stack">
                    <strong>{formatRunContext(run)}</strong>
                    <span>{formatDatasetName(run.dataset_name)}</span>
                  </div>
                </td>
                <td>
                  <div className="table-cell__stack">
                    <strong>{formatScore(run.average_score)}</strong>
                    <span>{run.step_count} steps</span>
                  </div>
                </td>
                <td>
                  <div className="artifact-strip">
                    {artifactKinds.length === 0 ? <span>pending</span> : artifactKinds.map((kind) => <span key={kind}>{kind}</span>)}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function JobsRail({
  jobs,
  selectedJobId,
  onSelect,
  onCancel,
  limit,
}: {
  jobs: JobSummary[];
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
  onCancel: (jobId: string) => void;
  limit?: number;
}) {
  const visibleJobs = limit ? jobs.slice(0, limit) : jobs;

  return (
    <div className="jobs-rail">
      {visibleJobs.length === 0 ? (
        <div className="empty-block">No jobs yet.</div>
      ) : (
        visibleJobs.map((job) => {
          const progress =
            job.progress_current !== null && job.progress_total !== null && job.progress_total > 0
              ? Math.min(100, (job.progress_current / job.progress_total) * 100)
              : null;

          return (
            <article
              key={job.job_id}
              className={`job-card ${selectedJobId === job.job_id ? "is-active" : ""} is-${job.status}`}
            >
              <div className="job-card__header">
                <strong>{job.runner_id.replace(/_/g, " ")}</strong>
                <span className={`status-chip is-${job.status}`}>{formatStatusLabel(job.status)}</span>
              </div>
              <div className="job-card__headline" title={job.run_id ?? undefined}>
                {job.run_id ? formatCompactRunId(job.run_id) : "Awaiting run id"}
              </div>
              <div className="job-card__meta-row">
                <span>{job.run_id ?? "run id pending"}</span>
                <span>{job.phase ? formatStatusLabel(job.phase) : "queued"}</span>
                <span>{formatRelativeTime(job.submitted_at)}</span>
              </div>
              {progress !== null ? (
                <div className="job-progress">
                  <div className="job-progress__fill" style={{ width: `${Math.max(4, progress)}%` }} />
                </div>
              ) : null}
              <div className="job-card__meta-row">
                <span>{job.progress_label ?? "no live label"}</span>
                <span>{job.average_score !== null ? `score ${formatScore(job.average_score)}` : "—"}</span>
              </div>
              {job.error_message ? <div className="job-card__error">{job.error_message}</div> : null}
              <div className="job-card__actions">
                <button className="text-button" onClick={() => onSelect(job.job_id)} type="button">
                  inspect
                </button>
                {job.status === "queued" || job.status === "running" ? (
                  <button className="ghost-button ghost-button--small" onClick={() => onCancel(job.job_id)} type="button">
                    cancel
                  </button>
                ) : null}
              </div>
            </article>
          );
        })
      )}
    </div>
  );
}

export function ChallengeMetricChart({ detail }: { detail: RunDetail }) {
  const data = Object.entries(detail.challenge_metrics)
    .filter(([key]) => key !== "average_score")
    .map(([key, value]) => ({
      key,
      label: value.display_name,
      value: value.value ?? 0,
    }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" stroke="rgba(42, 52, 48, 0.12)" />
        <XAxis dataKey="label" tick={{ fill: "#51605a", fontSize: 12 }} />
        <YAxis tick={{ fill: "#51605a", fontSize: 12 }} />
        <Tooltip />
        <Bar dataKey="value" fill="#264a3f" radius={[10, 10, 4, 4]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function PlaybackScene({
  playback,
  stepIndex,
}: {
  playback: PlaybackPayload;
  stepIndex: number;
}) {
  const payload = playback.payload;
  const buildings = payload.buildings ?? [];
  const media = payload.media ?? {};
  const frameStride = media.frame_stride ?? 1;
  const frameIndex = media.frames?.length
    ? Math.min(media.frames.length - 1, Math.floor(stepIndex / Math.max(frameStride, 1)))
    : null;
  const framePath = frameIndex === null ? media.poster_path : media.frames?.[frameIndex];
  const imageUrl = artifactUrl(framePath ?? media.poster_path ?? null);
  const buildingCards: Array<{
    name: string;
    battery: number;
    net: number;
    solar: number;
    outage: number;
  }> = buildings.map((building: Record<string, any>) => {
    const battery = building.series?.battery_soc?.[stepIndex] ?? 0;
    const net = building.series?.net_electricity_consumption?.[stepIndex] ?? 0;
    const solar = building.series?.solar_generation?.[stepIndex] ?? 0;
    const outage = building.series?.power_outage?.[stepIndex] ?? 0;
    return { name: building.name, battery, net, solar, outage };
  });

  return (
    <div className="playback-scene">
      <div className="scene-visual">
        <div className="scene-visual__frame">
          {imageUrl ? <img alt="CityLearn render" src={imageUrl} /> : <div className="scene-placeholder">No render captured.</div>}
        </div>
        <div className="scene-visual__meta">
          <span>{playback.mode === "full" ? "full capture" : "preview capture"}</span>
          <span>{buildingCards.length} buildings</span>
          <span>step {stepIndex}</span>
        </div>
      </div>
      <div className="scene-schematic">
        {buildingCards.map((building) => (
          <div key={building.name} className="building-chip">
            <div className="building-chip__top">
              <strong>{building.name}</strong>
              <span className={`status-chip ${building.outage ? "is-failed" : "is-succeeded"}`}>
                {building.outage ? "outage" : "online"}
              </span>
            </div>
            <div className="battery-rail">
              <div className="battery-rail__fill" style={{ width: `${Math.max(4, Math.min(100, building.battery))}%` }} />
            </div>
            <div className="building-chip__stats">
              <span>soc {formatScore(building.battery)}</span>
              <span>net {formatScore(building.net)}</span>
              <span>pv {formatScore(building.solar)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TimeseriesPanel({
  playback,
  stepIndex,
  selectedBuildingIndex,
  onSelectBuilding,
}: {
  playback: PlaybackPayload;
  stepIndex: number;
  selectedBuildingIndex: number;
  onSelectBuilding: (index: number) => void;
}) {
  const payload = playback.payload;
  const timestamps: string[] = payload.timestamps ?? [];
  const districtNet: number[] = payload.district?.net_electricity_consumption ?? [];
  const buildings = payload.buildings ?? [];
  const selectedBuilding = buildings[selectedBuildingIndex] ?? buildings[0];
  const chartData = timestamps.map((timestamp, index) => ({
    timestamp: timestamp.slice(5, 16).replace("T", " "),
    district: districtNet[index] ?? 0,
    battery: selectedBuilding?.series?.battery_soc?.[index] ?? 0,
    indoor: selectedBuilding?.series?.indoor_temperature?.[index] ?? 0,
    load: selectedBuilding?.series?.net_electricity_consumption?.[index] ?? 0,
    active: index === stepIndex ? 1 : 0,
  }));

  return (
    <div className="panel panel--charts">
      <div className="panel__header-stack">
        <div className="panel__title">Signals</div>
        <div className="building-selector">
          {buildings.map((building: Record<string, any>, index: number) => (
            <button
              key={building.name}
              className={`compare-pill ${index === selectedBuildingIndex ? "is-active" : ""}`}
              onClick={() => onSelectBuilding(index)}
              type="button"
            >
              {building.name}
            </button>
          ))}
        </div>
      </div>
      <div className="chart-grid">
        <div className="chart-frame">
          <div className="panel__title">District load / battery</div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData} margin={{ top: 8, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(42, 52, 48, 0.12)" />
              <XAxis dataKey="timestamp" tick={false} />
              <YAxis tick={{ fill: "#51605a", fontSize: 12 }} />
              <Tooltip />
              <Line type="monotone" dataKey="district" stroke="#1e4036" strokeWidth={2.25} dot={false} />
              <Line type="monotone" dataKey="battery" stroke="#b56744" strokeWidth={2.25} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-frame">
          <div className="panel__title">{selectedBuilding?.name ?? "Building"} load / indoor</div>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={chartData} margin={{ top: 8, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(42, 52, 48, 0.12)" />
              <XAxis dataKey="timestamp" tick={false} />
              <YAxis tick={{ fill: "#51605a", fontSize: 12 }} />
              <Tooltip />
              <Area type="monotone" dataKey="indoor" stroke="#6c8259" fill="rgba(108, 130, 89, 0.18)" />
              <Area type="monotone" dataKey="load" stroke="#365b6b" fill="rgba(54, 91, 107, 0.14)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export function TraceTable({
  frames,
  stepIndex,
  windowSize = 16,
}: {
  frames: PlaybackPayload["trace_frames"];
  stepIndex: number;
  windowSize?: number;
}) {
  if (frames.length === 0) {
    return <div className="empty-block">No trace frames yet.</div>;
  }

  const currentIndex = Math.max(
    0,
    frames.findIndex((frame) => frame.step === stepIndex),
  );
  const tentativeStart = Math.max(0, currentIndex - Math.floor(windowSize / 2));
  const tentativeEnd = Math.min(frames.length, tentativeStart + windowSize);
  const start = Math.max(0, tentativeEnd - windowSize);
  const visibleFrames = frames.slice(start, tentativeEnd);

  return (
    <div className="trace-table">
      {visibleFrames.map((frame) => (
        <div key={frame.step} className={`trace-row ${frame.step === stepIndex ? "is-active" : ""}`}>
          <div className="trace-row__step">step {frame.step}</div>
          <div className="trace-row__reward">reward {formatScore(frame.rewards[0] ?? null)}</div>
          <div className="trace-row__actions">
            {frame.actions[0]?.slice(0, 4).map((value, index) => (
              <span key={index}>{value.toFixed(2)}</span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export function CompareBars({ runs }: { runs: RunSummary[] }) {
  const data = runs.map((run) => ({
    name: formatRunTitle(run),
    score: run.average_score ?? 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(240, runs.length * 56)}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 8, left: 36, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" stroke="rgba(42, 52, 48, 0.12)" />
        <XAxis type="number" tick={{ fill: "#51605a", fontSize: 12 }} />
        <YAxis dataKey="name" type="category" tick={{ fill: "#51605a", fontSize: 12 }} width={132} />
        <Tooltip />
        <Bar dataKey="score" fill="#1e4036" radius={[0, 10, 10, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

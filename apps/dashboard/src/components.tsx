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
import { artifactUrl, formatMetricLabel, formatRelativeTime, formatScore } from "./lib/format";

export function SectionHeader({
  eyebrow,
  title,
  copy,
  action,
}: {
  eyebrow: string;
  title: string;
  copy: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-header">
      <div>
        <div className="section-header__eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        <p>{copy}</p>
      </div>
      {action}
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
            <span className={runner.launchable ? "is-live" : "is-muted"}>
              {runner.launchable ? "launchable" : "contract only"}
            </span>
          </div>
          <h3>{runner.label}</h3>
          <p>{runner.description}</p>
        </button>
      ))}
    </div>
  );
}

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="table-shell">
      <table className="runs-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Method</th>
            <th>Split</th>
            <th>Score</th>
            <th>Generated</th>
            <th>Artifacts</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id}>
              <td>
                <Link to={`/runs/${run.run_id}`}>{run.run_id}</Link>
              </td>
              <td>{run.algorithm.toUpperCase()} • {run.variant}</td>
              <td>{run.split}</td>
              <td>{formatScore(run.average_score)}</td>
              <td>{formatRelativeTime(run.generated_at)}</td>
              <td>
                <div className="artifact-strip">
                  {Boolean(run.artifacts.simulation_export) ? <span>simulation</span> : null}
                  {Boolean(run.artifacts.playback) ? <span>playback</span> : null}
                  {Boolean(run.artifacts.gif) ? <span>gif</span> : null}
                </div>
              </td>
            </tr>
          ))}
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
}: {
  jobs: JobSummary[];
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
  onCancel: (jobId: string) => void;
}) {
  return (
    <div className="jobs-rail">
      {jobs.length === 0 ? (
        <div className="empty-block">No jobs yet</div>
      ) : (
        jobs.map((job) => (
          <div
            key={job.job_id}
            className={`job-card job-card--${job.status} ${selectedJobId === job.job_id ? "is-active" : ""}`}
            onClick={() => onSelect(job.job_id)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect(job.job_id);
              }
            }}
            role="button"
            tabIndex={0}
          >
            <div className="job-card__header">
              <strong>{job.runner_id}</strong>
              <span>{job.status}</span>
            </div>
            <div className="job-card__meta">submitted {formatRelativeTime(job.submitted_at)}</div>
            {job.run_id ? <div className="job-card__meta">run {job.run_id}</div> : null}
            {job.average_score !== null ? (
              <div className="job-card__score">score {formatScore(job.average_score)}</div>
            ) : null}
            {job.error_message ? <div className="job-card__error">{job.error_message}</div> : null}
            {job.status === "queued" || job.status === "running" ? (
              <button className="ghost-button" onClick={() => onCancel(job.job_id)} type="button">
                cancel
              </button>
            ) : null}
          </div>
        ))
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
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(61, 70, 64, 0.12)" />
        <XAxis dataKey="label" tick={{ fill: "#536059", fontSize: 12 }} />
        <YAxis tick={{ fill: "#536059", fontSize: 12 }} />
        <Tooltip />
        <Bar dataKey="value" fill="#3f7059" radius={[12, 12, 4, 4]} />
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
        {imageUrl ? <img alt="CityLearn render" src={imageUrl} /> : <div className="scene-placeholder">no render captured</div>}
      </div>
      <div className="scene-schematic">
        {buildingCards.map((building) => (
          <div key={building.name} className="building-chip">
            <div className="building-chip__top">
              <strong>{building.name}</strong>
              <span>{building.outage ? "outage" : "online"}</span>
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
        <div className="panel__title">district and building traces</div>
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
      <div className="panel">
        <div className="panel__title">district load and selected battery state</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(61, 70, 64, 0.12)" />
            <XAxis dataKey="timestamp" tick={false} />
            <YAxis tick={{ fill: "#536059", fontSize: 12 }} />
            <Tooltip />
            <Line type="monotone" dataKey="district" stroke="#123629" strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="battery" stroke="#cc6e43" strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="panel">
        <div className="panel__title">selected building load and indoor temperature</div>
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(61, 70, 64, 0.12)" />
            <XAxis dataKey="timestamp" tick={false} />
            <YAxis tick={{ fill: "#536059", fontSize: 12 }} />
            <Tooltip />
            <Area type="monotone" dataKey="indoor" stroke="#7a915a" fill="rgba(122, 145, 90, 0.22)" />
            <Area type="monotone" dataKey="load" stroke="#2d5267" fill="rgba(45, 82, 103, 0.16)" />
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
}: {
  frames: PlaybackPayload["trace_frames"];
  stepIndex: number;
}) {
  return (
    <div className="trace-table">
      {frames.map((frame) => (
        <div key={frame.step} className={`trace-row ${frame.step === stepIndex ? "is-active" : ""}`}>
          <div>step {frame.step}</div>
          <div>reward {formatScore(frame.rewards[0] ?? null)}</div>
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
    name: run.run_id.slice(0, 18),
    score: run.average_score ?? 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(61, 70, 64, 0.12)" />
        <XAxis dataKey="name" tick={{ fill: "#536059", fontSize: 12 }} />
        <YAxis tick={{ fill: "#536059", fontSize: 12 }} />
        <Tooltip />
        <Bar dataKey="score" fill="#1c4c3c" radius={[12, 12, 4, 4]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

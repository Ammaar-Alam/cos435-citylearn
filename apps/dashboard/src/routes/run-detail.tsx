import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import {
  ChallengeMetricChart,
  MetricCard,
  PlaybackScene,
  SectionHeader,
  TimeseriesPanel,
  TraceTable,
} from "../components";
import { fetchPlayback, fetchRunDetail } from "../lib/api";
import { OFFICIAL_BENCHMARK } from "../lib/benchmark";
import {
  artifactUrl,
  formatDatasetName,
  formatMetricLabel,
  formatRelativeTime,
  formatRunContext,
  formatRunTitle,
  formatScore,
  formatSignedScore,
  getRunArtifactKinds,
  getRunArtifactPath,
} from "../lib/format";
import { useInterval } from "../lib/useInterval";

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const detailQuery = useQuery({ queryKey: ["run", runId], queryFn: () => fetchRunDetail(runId) });
  const playbackQuery = useQuery({
    queryKey: ["playback", runId],
    queryFn: () => fetchPlayback(runId),
  });

  const [isPlaying, setIsPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [selectedBuildingIndex, setSelectedBuildingIndex] = useState(0);

  useInterval(() => {
    if (!isPlaying || !playbackQuery.data?.payload?.time_steps) {
      return;
    }

    setStepIndex((current) => {
      const total = playbackQuery.data?.payload?.time_steps ?? 0;
      return current >= total - 1 ? 0 : current + 1;
    });
  }, isPlaying ? 400 : null);

  const detail = detailQuery.data;
  const playback = playbackQuery.data;

  useEffect(() => {
    setStepIndex(0);
    setSelectedBuildingIndex(0);
    setIsPlaying(false);
  }, [runId]);

  useEffect(() => {
    const buildingCount = playback?.payload?.buildings?.length ?? playback?.building_names?.length ?? 0;
    if (selectedBuildingIndex >= buildingCount) {
      setSelectedBuildingIndex(0);
    }
  }, [playback?.building_names?.length, playback?.payload?.buildings?.length, selectedBuildingIndex]);

  const scoreCards = useMemo(() => {
    if (!detail) {
      return [];
    }

    return Object.entries(detail.challenge_metrics)
      .filter(([key]) => key !== "average_score")
      .slice(0, 4)
      .map(([key, metric]) => ({
        key,
        label: metric.display_name || formatMetricLabel(key),
        value: formatScore(metric.value),
      }));
  }, [detail]);

  if (!detail || !playback) {
    return (
      <div className="page-stack page-stack--run-detail">
        <div className="panel">Loading run detail…</div>
      </div>
    );
  }

  const summary = detail.summary;
  const totalSteps = playback.total_steps ?? summary.step_count;
  const posterPath = playback.payload?.media?.poster_path ?? getRunArtifactPath(summary, "poster");
  const districtRows = Object.entries(detail.district_kpis).slice(0, 8);
  const baselineGap = summary.average_score === null ? null : summary.average_score - OFFICIAL_BENCHMARK.baselinePrivate;
  const winnerGap = summary.average_score === null ? null : summary.average_score - OFFICIAL_BENCHMARK.winnerPrivate;

  const artifactRows = [
    { label: "poster", path: posterPath },
    { label: "gif playback", path: playback.payload?.media?.gif_path ?? getRunArtifactPath(summary, "gif") },
    { label: "playback payload", path: getRunArtifactPath(summary, "playback") },
    { label: "simulation export", path: getRunArtifactPath(summary, "simulation_export") },
  ]
    .filter((row): row is { label: string; path: string } => Boolean(row.path))
    .filter((row, index, array) => array.findIndex((candidate) => candidate.path === row.path) === index);

  return (
    <div className="page-stack page-stack--run-detail">
      <section className="page-header">
        <div className="page-header__body">
          <div className="page-header__eyebrow">Run detail</div>
          <h1>{formatRunTitle(summary)}</h1>
          <p>{`${formatRunContext(summary)} · ${formatDatasetName(summary.dataset_name)}`}</p>
        </div>
        <div className="page-header__actions">
          <Link className="primary-button" to={`/compare?runIds=${summary.run_id}`}>
            compare
          </Link>
          <Link className="ghost-button" to="/runs">
            archive
          </Link>
        </div>
      </section>

      <section className="run-detail-hero">
        <article className="panel selected-run-card selected-run-card--detail">
          <div className="selected-run-card__media">
            {posterPath ? (
              <img alt="Run poster" src={artifactUrl(posterPath) ?? ""} />
            ) : (
              <div className="scene-placeholder">No poster captured for this run.</div>
            )}
          </div>
          <div className="selected-run-card__body">
            <div className="panel__title">{formatDatasetName(summary.dataset_name)}</div>
            <h2>{formatRunContext(summary)}</h2>
            <p>{summary.run_id}</p>
            <div className="metric-row metric-row--artifact">
              <MetricCard label="score" value={formatScore(summary.average_score)} tone="warm" hint="lower is better" />
              <MetricCard label="vs RBC private" value={formatSignedScore(baselineGap)} />
              <MetricCard label="vs winner" value={formatSignedScore(winnerGap)} />
              <MetricCard label="steps" value={String(summary.step_count)} />
            </div>
            {scoreCards.length > 0 ? (
              <div className="artifact-strip">
                {scoreCards.map((card) => (
                  <span key={card.key}>
                    {card.label}: {card.value}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        </article>

        <article className="panel panel--quiet">
          <SectionHeader eyebrow="Evidence" title="Run facts" />
          <div className="fact-list">
            <div className="fact-list__row">
              <span>dataset</span>
              <strong>{formatDatasetName(summary.dataset_name)}</strong>
            </div>
            <div className="fact-list__row">
              <span>split</span>
              <strong>{formatMetricLabel(summary.split)}</strong>
            </div>
            <div className="fact-list__row">
              <span>seed</span>
              <strong>{summary.seed}</strong>
            </div>
            <div className="fact-list__row">
              <span>generated</span>
              <strong>{formatRelativeTime(summary.generated_at)}</strong>
            </div>
            <div className="fact-list__row">
              <span>stored steps</span>
              <strong>{playback.stored_steps}</strong>
            </div>
            <div className="fact-list__row">
              <span>artifacts</span>
              <strong>{getRunArtifactKinds(summary).join(" • ") || "pending"}</strong>
            </div>
          </div>
          <div className="artifact-list">
            {artifactRows.length > 0 ? (
              artifactRows.map((artifact) => (
                <a key={artifact.path} className="artifact-row" href={artifactUrl(artifact.path) ?? "#"} rel="noreferrer" target="_blank">
                  <strong>{artifact.label}</strong>
                  <span>{artifact.path.split("/").filter(Boolean).pop() ?? artifact.path}</span>
                </a>
              ))
            ) : (
              <div className="empty-block">No addressable files were attached to this run.</div>
            )}
          </div>
        </article>
      </section>

      <article className="panel">
        <SectionHeader
          eyebrow="Playback"
          title="Step viewer"
          action={
            <div className="playback-controls">
              <button className="ghost-button ghost-button--small" onClick={() => setIsPlaying((value) => !value)} type="button">
                {isPlaying ? "pause" : "play"}
              </button>
              <input
                className="playback-slider"
                max={Math.max(totalSteps - 1, 0)}
                min={0}
                onChange={(event) => setStepIndex(Number(event.target.value))}
                type="range"
                value={stepIndex}
              />
              <span>step {stepIndex}</span>
            </div>
          }
        />
        <PlaybackScene playback={playback} stepIndex={stepIndex} />
      </article>

      <TimeseriesPanel
        playback={playback}
        stepIndex={stepIndex}
        selectedBuildingIndex={selectedBuildingIndex}
        onSelectBuilding={setSelectedBuildingIndex}
      />

      <section className="content-grid">
        <article className="panel panel--quiet">
          <SectionHeader eyebrow="Challenge metrics" title="Score anatomy" />
          <ChallengeMetricChart detail={detail} />
        </article>
        <article className="panel panel--quiet">
          <SectionHeader eyebrow="District KPI slice" title="District KPIs" />
          <div className="fact-list">
            {districtRows.map(([key, value]) => (
              <div key={key} className="fact-list__row">
                <span>{formatMetricLabel(key)}</span>
                <strong>{formatScore(value)}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>

      <article className="panel panel--quiet">
        <SectionHeader
          eyebrow="Trace"
          title={playback.mode === "full" ? "Recorded trace" : "Preview trace"}
        />
        <TraceTable frames={playback.trace_frames} stepIndex={stepIndex} windowSize={12} />
      </article>
    </div>
  );
}

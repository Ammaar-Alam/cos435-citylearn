import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import {
  ChallengeMetricChart,
  MetricCard,
  PlaybackScene,
  SectionHeader,
  TimeseriesPanel,
  TraceTable,
} from "../components";
import { fetchPlayback, fetchRunDetail } from "../lib/api";
import { formatMetricLabel, formatScore } from "../lib/format";
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
  const buildingNames = playback?.payload?.building_names ?? playback?.building_names ?? [];

  useEffect(() => {
    if (selectedBuildingIndex >= buildingNames.length) {
      setSelectedBuildingIndex(0);
    }
  }, [buildingNames.length, selectedBuildingIndex]);
  const scoreCards = useMemo(() => {
    if (!detail) {
      return [];
    }

    return Object.entries(detail.challenge_metrics)
      .slice(0, 4)
      .map(([key, metric]) => ({
        key,
        label: metric.display_name,
        value: formatScore(metric.value),
      }));
  }, [detail]);

  if (!detail || !playback) {
    return <div className="page-stack"><div className="panel">Loading run detail…</div></div>;
  }

  const totalSteps = playback.total_steps ?? detail.summary.step_count;
  const districtRows = Object.entries(detail.district_kpis).slice(0, 8);

  return (
    <div className="page-stack">
      <section className="panel panel--feature">
        <SectionHeader
          eyebrow={detail.summary.algorithm.toUpperCase()}
          title={detail.summary.run_id}
          copy="The literal scene and the data-first schematic stay in sync. This view is designed for actual inspection and for capturing slide-ready snapshots."
        />
        <div className="metric-row">
          <MetricCard label="average score" value={formatScore(detail.summary.average_score)} tone="warm" />
          <MetricCard label="split" value={detail.summary.split} />
          <MetricCard label="dataset" value={detail.summary.dataset_name.replace(/_/g, " ")} />
          <MetricCard label="steps" value={String(detail.summary.step_count)} />
          {scoreCards.map((card) => (
            <MetricCard key={card.key} label={card.label} value={card.value} tone="mint" />
          ))}
        </div>
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="playback"
          title="Literal scene + schematic"
          copy="This is driven from the saved playback payload and media artifacts, not from a silent rerun."
          action={
            <div className="playback-controls">
              <button className="ghost-button" onClick={() => setIsPlaying((value) => !value)} type="button">
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
      </section>

      <TimeseriesPanel
        playback={playback}
        stepIndex={stepIndex}
        selectedBuildingIndex={selectedBuildingIndex}
        onSelectBuilding={setSelectedBuildingIndex}
      />

      <section className="content-grid">
        <div className="panel">
          <SectionHeader
            eyebrow="challenge metrics"
            title="Score anatomy"
            copy="These are the same challenge metrics exported in the benchmark run directory."
          />
          <ChallengeMetricChart detail={detail} />
        </div>
        <div className="panel">
          <SectionHeader
            eyebrow="district kpis"
            title="Raw KPI slice"
            copy="Keep the full table in the JSON. This panel is just a quick inspection slice."
          />
          <div className="env-list">
            {districtRows.map(([key, value]) => (
              <div key={key} className="env-list__row">
                <span>{formatMetricLabel(key)}</span>
                <strong>{formatScore(value)}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="trace"
          title={playback.mode === "full" ? "Recorded action trace" : "Preview trace"}
          copy={
            playback.mode === "full"
              ? playback.truncated
                ? "The playback payload is richer than the saved trace, so this table shows the stored slice only."
                : "The trace is synced with the richer playback payload."
              : "This run only has the lightweight rollout preview artifact."
          }
        />
        <TraceTable frames={playback.trace_frames} stepIndex={stepIndex} />
      </section>
    </div>
  );
}

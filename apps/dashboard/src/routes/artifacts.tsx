import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  evaluateArtifact,
  fetchArtifactPlayback,
  fetchArtifacts,
  fetchRunners,
  importArtifact,
} from "../lib/api";
import { formatRelativeTime } from "../lib/format";
import { JobsRail, PlaybackScene, SectionHeader, TimeseriesPanel, TraceTable } from "../components";

export function ArtifactsPage() {
  const queryClient = useQueryClient();
  const [artifactKind, setArtifactKind] = useState<"checkpoint" | "run_bundle" | "simulation_bundle">("checkpoint");
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [selectedRunnerId, setSelectedRunnerId] = useState<string>("");
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedBuildingIndex, setSelectedBuildingIndex] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);
  const artifactsQuery = useQuery({ queryKey: ["artifacts"], queryFn: fetchArtifacts });
  const runnersQuery = useQuery({ queryKey: ["runners"], queryFn: fetchRunners });
  const selectedArtifact = useMemo(
    () => artifactsQuery.data?.find((artifact) => artifact.artifact_id === selectedArtifactId) ?? artifactsQuery.data?.[0] ?? null,
    [artifactsQuery.data, selectedArtifactId],
  );
  const playbackQuery = useQuery({
    queryKey: ["artifact-playback", selectedArtifact?.artifact_id],
    queryFn: () => fetchArtifactPlayback(selectedArtifact!.artifact_id),
    enabled: Boolean(selectedArtifact?.artifact_id && selectedArtifact?.playback_path),
  });

  useEffect(() => {
    if (!selectedArtifact && artifactsQuery.data?.[0]) {
      setSelectedArtifactId(artifactsQuery.data[0].artifact_id);
    }
  }, [artifactsQuery.data, selectedArtifact]);

  useEffect(() => {
    if (selectedBuildingIndex >= (playbackQuery.data?.building_names?.length ?? 0)) {
      setSelectedBuildingIndex(0);
    }
  }, [playbackQuery.data?.building_names?.length, selectedBuildingIndex]);

  useEffect(() => {
    setStepIndex(0);
  }, [selectedArtifact?.artifact_id]);

  const uploadMutation = useMutation({
    mutationFn: importArtifact,
    onSuccess: (artifact) => {
      queryClient.invalidateQueries({ queryKey: ["artifacts"] });
      setSelectedArtifactId(artifact.artifact_id);
      setSelectedFile(null);
      setLabel("");
      setNotes("");
    },
  });

  const evaluateMutation = useMutation({
    mutationFn: (artifactId: string) =>
      evaluateArtifact(artifactId, {
        seed: 0,
        split: "public_dev",
        trace_limit: 128,
        capture_render_frames: true,
        max_render_frames: 72,
        render_frame_width: 960,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  function handleSubmit(): void {
    if (!selectedFile) {
      return;
    }

    const payload = new FormData();
    payload.append("file", selectedFile);
    payload.append("artifact_kind", artifactKind);
    payload.append("label", label);
    if (notes.trim()) {
      payload.append("notes", notes.trim());
    }
    if (artifactKind === "checkpoint" && selectedRunnerId) {
      payload.append("runner_id", selectedRunnerId);
    }
    uploadMutation.mutate(payload);
  }

  const checkpointRunners = (runnersQuery.data ?? []).filter((runner) => runner.supports_checkpoint_eval);
  const playback = playbackQuery.data;

  return (
    <div className="page-stack">
      <section className="split-grid">
        <div className="panel panel--feature">
          <SectionHeader
            eyebrow="imports"
            title="Bring outside artifacts into the same dashboard"
            copy="Playback payloads can be inspected immediately. Checkpoints can be registered now, but this branch still does not have a checkpoint evaluator for SAC or PPO."
          />
          <div className="form-grid">
            <label>
              artifact kind
              <select value={artifactKind} onChange={(event) => setArtifactKind(event.target.value as typeof artifactKind)}>
                <option value="checkpoint">checkpoint</option>
                <option value="run_bundle">run bundle</option>
                <option value="simulation_bundle">simulation bundle</option>
              </select>
            </label>
            <label>
              label
              <input onChange={(event) => setLabel(event.target.value)} placeholder="optional display name" value={label} />
            </label>
            <label className="form-grid__wide">
              file
              <input
                onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                type="file"
              />
            </label>
            {artifactKind === "checkpoint" ? (
              <label className="form-grid__wide">
                checkpoint runner
                <select value={selectedRunnerId} onChange={(event) => setSelectedRunnerId(event.target.value)}>
                  <option value="">select runner</option>
                  {checkpointRunners.map((runner) => (
                    <option key={runner.runner_id} value={runner.runner_id}>
                      {runner.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="form-grid__wide">
              notes
              <textarea onChange={(event) => setNotes(event.target.value)} placeholder="what this artifact is for" rows={3} value={notes} />
            </label>
          </div>
          <div className="inline-actions">
            <button className="primary-button" disabled={!selectedFile || uploadMutation.isPending} onClick={handleSubmit} type="button">
              {uploadMutation.isPending ? "importing…" : "import artifact"}
            </button>
            <div className="muted-copy">
              {artifactKind === "checkpoint" && checkpointRunners.length === 0
                ? "No local checkpoint-evaluable runners exist yet in this branch. You can still register the checkpoint so it is ready once inference support lands."
                : "Playback JSON imports become inspectable immediately if they already contain a saved trace."}
            </div>
          </div>
        </div>

        <div className="panel">
          <SectionHeader
            eyebrow="registry"
            title="Imported artifacts"
            copy="Artifacts stay on disk under the dashboard registry so they can be re-used without rebuilding the underlying run folders."
          />
          <div className="artifact-list">
            {(artifactsQuery.data ?? []).length === 0 ? (
              <div className="empty-block">No imported artifacts yet.</div>
            ) : (
              (artifactsQuery.data ?? []).map((artifact) => (
                <button
                  key={artifact.artifact_id}
                  className={`artifact-card ${selectedArtifact?.artifact_id === artifact.artifact_id ? "is-active" : ""}`}
                  onClick={() => setSelectedArtifactId(artifact.artifact_id)}
                  type="button"
                >
                  <div className="artifact-card__header">
                    <strong>{artifact.label}</strong>
                    <span>{artifact.status}</span>
                  </div>
                  <div className="artifact-card__meta">{artifact.artifact_kind.replace(/_/g, " ")}</div>
                  <div className="artifact-card__meta">{artifact.algorithm}</div>
                  <div className="artifact-card__meta">{formatRelativeTime(artifact.imported_at)}</div>
                </button>
              ))
            )}
          </div>
        </div>
      </section>

      {selectedArtifact ? (
        <section className="panel">
          <SectionHeader
            eyebrow="selected artifact"
            title={selectedArtifact.label}
            copy={`${selectedArtifact.artifact_kind.replace(/_/g, " ")} • ${selectedArtifact.source_filename}`}
            action={
              selectedArtifact.evaluable ? (
                <button
                  className="primary-button"
                  disabled={evaluateMutation.isPending}
                  onClick={() => evaluateMutation.mutate(selectedArtifact.artifact_id)}
                  type="button"
                >
                  {evaluateMutation.isPending ? "starting eval…" : "evaluate locally"}
                </button>
              ) : null
            }
          />
          <div className="metric-row metric-row--artifact">
            <div className="metric-card">
              <div className="metric-card__label">artifact id</div>
              <div className="metric-card__hint">{selectedArtifact.artifact_id}</div>
            </div>
            <div className="metric-card">
              <div className="metric-card__label">runner binding</div>
              <div className="metric-card__hint">{selectedArtifact.runner_id ?? "none"}</div>
            </div>
            <div className="metric-card">
              <div className="metric-card__label">playback</div>
              <div className="metric-card__hint">{selectedArtifact.playback_path ?? "not present"}</div>
            </div>
            <div className="metric-card">
              <div className="metric-card__label">simulation data</div>
              <div className="metric-card__hint">{selectedArtifact.simulation_dir ?? "not present"}</div>
            </div>
          </div>
        </section>
      ) : null}

      {playback ? (
        <>
          <section className="panel">
            <SectionHeader
              eyebrow="playback"
              title="Imported playback preview"
              copy="If the imported file already contains a saved playback payload, you can inspect it here before deciding whether it needs a fresh local evaluation."
              action={
                <div className="playback-controls">
                  <input
                    className="playback-slider"
                    max={Math.max(playback.stored_steps - 1, 0)}
                    min={0}
                    onChange={(event) => setStepIndex(Number(event.target.value))}
                    type="range"
                    value={Math.min(stepIndex, Math.max(playback.stored_steps - 1, 0))}
                  />
                  <span>step {Math.min(stepIndex, Math.max(playback.stored_steps - 1, 0))}</span>
                </div>
              }
            />
            <PlaybackScene playback={playback} stepIndex={Math.min(stepIndex, Math.max(playback.stored_steps - 1, 0))} />
          </section>
          <TimeseriesPanel
            playback={playback}
            stepIndex={Math.min(stepIndex, Math.max(playback.stored_steps - 1, 0))}
            selectedBuildingIndex={selectedBuildingIndex}
            onSelectBuilding={setSelectedBuildingIndex}
          />
          <section className="panel">
            <SectionHeader
              eyebrow="trace"
              title="Imported trace"
              copy="This is the recorded action history from the uploaded playback payload."
            />
            <TraceTable frames={playback.trace_frames} stepIndex={Math.min(stepIndex, Math.max(playback.stored_steps - 1, 0))} />
          </section>
        </>
      ) : selectedArtifact ? (
        <section className="panel">
          <SectionHeader
            eyebrow="next action"
            title="No playback attached yet"
            copy="Checkpoint artifacts are still useful even before a local eval exists. Once the runner supports checkpoint evaluation, this page can launch an eval and then hand you off to the monitor."
          />
          {evaluateMutation.data ? (
            <Link className="primary-button" to="/monitor">
              open monitor
            </Link>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}

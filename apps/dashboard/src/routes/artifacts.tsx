import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  evaluateArtifact,
  fetchArtifact,
  fetchArtifactPlayback,
  fetchArtifacts,
  fetchRunners,
  importArtifact,
} from "../lib/api";
import { artifactUrl, basename, formatRelativeTime, formatStatusLabel } from "../lib/format";
import { MetricCard, PlaybackScene, SectionHeader, TimeseriesPanel, TraceTable } from "../components";

type ArtifactKind = "checkpoint" | "run_bundle" | "simulation_bundle";

export function ArtifactsPage() {
  const queryClient = useQueryClient();
  const [artifactKind, setArtifactKind] = useState<ArtifactKind>("run_bundle");
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [selectedRunnerId, setSelectedRunnerId] = useState("");
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedBuildingIndex, setSelectedBuildingIndex] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  const artifactsQuery = useQuery({ queryKey: ["artifacts"], queryFn: fetchArtifacts });
  const runnersQuery = useQuery({ queryKey: ["runners"], queryFn: fetchRunners });

  const selectedArtifact = useMemo(
    () =>
      artifactsQuery.data?.find((artifact) => artifact.artifact_id === selectedArtifactId) ??
      artifactsQuery.data?.[0] ??
      null,
    [artifactsQuery.data, selectedArtifactId],
  );

  const artifactDetailQuery = useQuery({
    queryKey: ["artifact", selectedArtifact?.artifact_id],
    queryFn: () => fetchArtifact(selectedArtifact!.artifact_id),
    enabled: Boolean(selectedArtifact?.artifact_id),
  });

  const playbackQuery = useQuery({
    queryKey: ["artifact-playback", selectedArtifact?.artifact_id],
    queryFn: () => fetchArtifactPlayback(selectedArtifact!.artifact_id),
    enabled: Boolean(selectedArtifact?.artifact_id && selectedArtifact?.playback_path),
    retry: false,
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
  const totalArtifacts = artifactsQuery.data?.length ?? 0;
  const playbackCount = (artifactsQuery.data ?? []).filter((artifact) => artifact.playback_path).length;
  const checkpointCount = (artifactsQuery.data ?? []).filter((artifact) => artifact.artifact_kind === "checkpoint").length;
  const evaluableCount = (artifactsQuery.data ?? []).filter((artifact) => artifact.evaluable).length;

  const selectedArtifactRows = useMemo(() => {
    if (!selectedArtifact) {
      return [];
    }

    const rows = [
      selectedArtifact.playback_path
        ? { label: "playback payload", path: selectedArtifact.playback_path }
        : null,
      selectedArtifact.simulation_dir
        ? { label: "simulation export", path: selectedArtifact.simulation_dir }
        : null,
    ].filter((row): row is { label: string; path: string } => Boolean(row));

    if (artifactDetailQuery.data?.file_path) {
      rows.unshift({ label: "registry file", path: artifactDetailQuery.data.file_path });
    }

    return rows.filter(
      (row, index, array) => array.findIndex((candidate) => candidate.path === row.path) === index,
    );
  }, [artifactDetailQuery.data?.file_path, selectedArtifact]);

  const readinessCopy =
    artifactKind === "checkpoint"
      ? checkpointRunners.length > 0
        ? "Bind a runner when you want to stage the next eval."
        : "Store the checkpoint now and bind it once the evaluator path lands."
      : "Run bundles and simulation exports can be inspected immediately.";

  return (
    <div className="page-stack page-stack--artifacts">
      <section className="page-header">
        <div className="page-header__body">
          <div className="page-header__eyebrow">Artifacts</div>
          <h1>Artifact registry</h1>
          <p>Register files, inspect playback, and queue local evaluation.</p>
        </div>
        <div className="page-header__actions">
          {evaluateMutation.data ? (
            <Link className="primary-button" to="/monitor">
              live monitor
            </Link>
          ) : null}
          <Link className="ghost-button" to="/runs">
            run archive
          </Link>
        </div>
      </section>

      <section className="artifacts-shell">
        <article className="panel panel--feature">
          <SectionHeader
            eyebrow="Intake"
            title="Register artifact"
          />
          <div className="metric-row metric-row--artifact">
            <MetricCard label="registry total" value={String(totalArtifacts)} />
            <MetricCard label="playback ready" value={String(playbackCount)} tone="mint" />
            <MetricCard label="checkpoint records" value={String(checkpointCount)} />
            <MetricCard label="eval ready" value={String(evaluableCount)} tone="warm" />
          </div>
          <div className="form-grid">
            <label>
              artifact kind
              <select value={artifactKind} onChange={(event) => setArtifactKind(event.target.value as ArtifactKind)}>
                <option value="run_bundle">run bundle</option>
                <option value="simulation_bundle">simulation bundle</option>
                <option value="checkpoint">checkpoint</option>
              </select>
            </label>
            <label>
              label
              <input onChange={(event) => setLabel(event.target.value)} placeholder="optional display name" value={label} />
            </label>
            <label className="form-grid__wide">
              file
              <input onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)} type="file" />
            </label>
            {artifactKind === "checkpoint" ? (
              <label className="form-grid__wide">
                runner binding
                <select value={selectedRunnerId} onChange={(event) => setSelectedRunnerId(event.target.value)}>
                  <option value="">store without binding</option>
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
              <textarea
                onChange={(event) => setNotes(event.target.value)}
                placeholder="optional intake note"
                rows={3}
                value={notes}
              />
            </label>
          </div>
          <div className="inline-actions">
            <button className="primary-button" disabled={!selectedFile || uploadMutation.isPending} onClick={handleSubmit} type="button">
              {uploadMutation.isPending ? "importing…" : "register artifact"}
            </button>
            <div className="muted-copy">{readinessCopy}</div>
          </div>
        </article>

        <div className="page-stack">
          <article className="panel panel--quiet">
            <SectionHeader
              eyebrow="Registry"
              title="Stored records"
            />
            <div className="artifact-list">
              {totalArtifacts === 0 ? (
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
                      <span className={`status-chip ${artifact.evaluable ? "is-succeeded" : "is-muted"}`}>
                        {artifact.evaluable ? "ready" : "stored"}
                      </span>
                    </div>
                    <div className="artifact-card__meta">{artifact.artifact_kind.replace(/_/g, " ")}</div>
                    <div className="artifact-card__meta">{artifact.algorithm}</div>
                    <div className="artifact-card__meta">{formatRelativeTime(artifact.imported_at)}</div>
                  </button>
                ))
              )}
            </div>
          </article>

          <article className="panel panel--quiet">
            <SectionHeader eyebrow="Readiness" title="Registry state" />
            <div className="fact-list">
              <div className="fact-list__row">
                <span>run bundles</span>
                <strong>{(artifactsQuery.data ?? []).filter((artifact) => artifact.artifact_kind === "run_bundle").length}</strong>
              </div>
              <div className="fact-list__row">
                <span>simulation bundles</span>
                <strong>{(artifactsQuery.data ?? []).filter((artifact) => artifact.artifact_kind === "simulation_bundle").length}</strong>
              </div>
              <div className="fact-list__row">
                <span>checkpoints</span>
                <strong>{checkpointCount}</strong>
              </div>
              <div className="fact-list__row">
                <span>playback attached</span>
                <strong>{playbackCount}</strong>
              </div>
              <div className="fact-list__row">
                <span>ready for eval</span>
                <strong>{evaluableCount}</strong>
              </div>
            </div>
          </article>
        </div>
      </section>

      {selectedArtifact ? (
        <article className="panel panel--quiet">
          <SectionHeader
            eyebrow="Selected artifact"
            title={selectedArtifact.label}
            copy={`${selectedArtifact.artifact_kind.replace(/_/g, " ")} · ${selectedArtifact.source_filename}`}
            action={
              selectedArtifact.evaluable ? (
                <button
                  className="primary-button"
                  disabled={evaluateMutation.isPending}
                  onClick={() => evaluateMutation.mutate(selectedArtifact.artifact_id)}
                  type="button"
                >
                  {evaluateMutation.isPending ? "starting eval…" : "evaluate"}
                </button>
              ) : null
            }
          />
          <section className="content-grid">
            <div className="page-stack">
              <div className="metric-row metric-row--artifact">
                <MetricCard label="status" value={formatStatusLabel(selectedArtifact.status)} />
                <MetricCard
                  label="runner binding"
                  value={selectedArtifact.runner_id ?? "none"}
                  hint={selectedArtifact.evaluable ? "evaluation path available" : "binding only"}
                />
                <MetricCard
                  label="playback"
                  value={selectedArtifact.playback_path ? "attached" : "missing"}
                  tone={selectedArtifact.playback_path ? "mint" : "neutral"}
                />
                <MetricCard label="imported" value={formatRelativeTime(selectedArtifact.imported_at)} />
              </div>
              {artifactDetailQuery.data?.notes ? (
                <div className="note-block">
                  <strong>Notes</strong>
                  <p>{artifactDetailQuery.data.notes}</p>
                </div>
              ) : null}
            </div>

            <div className="page-stack">
              <div className="artifact-list">
                {selectedArtifactRows.length > 0 ? (
                  selectedArtifactRows.map((row) => {
                    const href = artifactUrl(row.path);

                    if (href) {
                      return (
                        <a key={`${row.label}-${row.path}`} className="artifact-row" href={href} rel="noreferrer" target="_blank">
                          <strong>{row.label}</strong>
                          <span>{basename(row.path)}</span>
                        </a>
                      );
                    }

                    return (
                      <div key={`${row.label}-${row.path}`} className="artifact-row">
                        <strong>{row.label}</strong>
                        <span>{basename(row.path)}</span>
                      </div>
                    );
                  })
                ) : (
                  <div className="empty-block">No addressable files are attached yet.</div>
                )}
              </div>
              <div className="note-block">
                <strong>{selectedArtifact.evaluable ? "Eval path ready" : "Stored for later"}</strong>
                <p>
                  {selectedArtifact.evaluable
                    ? "Launching from here creates a fresh local run bundle."
                    : "This record stays addressable even without an evaluator path."}
                </p>
              </div>
            </div>
          </section>
        </article>
      ) : null}

      {playback ? (
        <>
          <article className="panel panel--quiet">
            <SectionHeader
              eyebrow="Playback"
              title="Playback preview"
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
          </article>

          <TimeseriesPanel
            playback={playback}
            stepIndex={Math.min(stepIndex, Math.max(playback.stored_steps - 1, 0))}
            selectedBuildingIndex={selectedBuildingIndex}
            onSelectBuilding={setSelectedBuildingIndex}
          />

          <article className="panel panel--quiet">
            <SectionHeader eyebrow="Trace" title="Stored trace" />
            <TraceTable
              frames={playback.trace_frames}
              stepIndex={Math.min(stepIndex, Math.max(playback.stored_steps - 1, 0))}
              windowSize={12}
            />
          </article>
        </>
      ) : selectedArtifact ? (
        <article className="panel panel--quiet">
          <SectionHeader eyebrow="Next step" title="No playback attached yet" />
          <div className="note-block">
            <strong>{selectedArtifact.artifact_kind === "checkpoint" ? "Checkpoint record" : "Registry record"}</strong>
            <p>
              {selectedArtifact.artifact_kind === "checkpoint"
                ? "Use the runner binding when you want to queue a local evaluation."
                : "This import is stored now and can be revisited later."}
            </p>
          </div>
        </article>
      ) : null}
    </div>
  );
}

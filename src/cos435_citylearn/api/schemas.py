from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ArtifactKind = Literal["checkpoint", "run_bundle", "simulation_bundle"]


class RunnerSummary(BaseModel):
    runner_id: str
    label: str
    algorithm: str
    variant: str
    description: str
    config_path: str
    eval_config_path: str
    launchable: bool
    supports_checkpoint_eval: bool = False


class LaunchJobRequest(BaseModel):
    runner_id: str = "rbc_builtin"
    artifact_id: str | None = None
    seed: int | None = Field(default=None, ge=0)
    split: str | None = None
    trace_limit: int | None = Field(default=None, ge=8, le=720)
    capture_render_frames: bool | None = None
    max_render_frames: int | None = Field(default=None, ge=8, le=180)
    render_frame_width: int | None = Field(default=None, ge=480, le=1600)


class JobSummary(BaseModel):
    job_id: str
    runner_id: str
    status: Literal["queued", "running", "succeeded", "failed", "orphaned", "cancelled"]
    phase: str | None = None
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    config_path: str
    eval_config_path: str
    run_id: str | None = None
    average_score: float | None = None
    error_message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_label: str | None = None
    heartbeat_at: str | None = None
    latest_preview_path: str | None = None


class RunSummary(BaseModel):
    run_id: str
    algorithm: str
    variant: str
    split: str
    seed: int
    dataset_name: str
    generated_at: str
    step_count: int
    average_score: float | None
    artifacts: dict[str, Any]


class RunDetail(BaseModel):
    summary: RunSummary
    challenge_metrics: dict[str, dict[str, float | str | None]]
    district_kpis: dict[str, float | None]
    manifest: dict[str, Any]


class PlaybackFrame(BaseModel):
    step: int
    actions: list[list[float]]
    rewards: list[float]
    terminated: bool


class PlaybackResponse(BaseModel):
    run_id: str
    mode: Literal["preview", "full"]
    total_steps: int
    stored_steps: int
    truncated: bool
    action_names: list[list[str]]
    building_names: list[str]
    offset: int
    limit: int
    trace_frames: list[PlaybackFrame]
    payload: dict[str, Any]


class JobArtifact(BaseModel):
    kind: str
    label: str
    path: str


class JobState(BaseModel):
    job_id: str
    job_kind: str
    status: str
    phase: str
    progress_current: int | None = None
    progress_total: int | None = None
    progress_label: str | None = None
    heartbeat_at: str
    latest_run_id: str | None = None
    latest_preview_path: str | None = None
    latest_checkpoint_id: str | None = None
    latest_log_offset: int | None = None
    error_message: str | None = None


class JobEvent(BaseModel):
    seq: int
    job_id: str
    event_type: str
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ArtifactSummary(BaseModel):
    artifact_id: str
    artifact_kind: ArtifactKind
    label: str
    source_filename: str
    imported_at: str
    algorithm: str
    runner_id: str | None = None
    status: str
    evaluable: bool
    playback_path: str | None = None
    simulation_dir: str | None = None


class ArtifactDetail(ArtifactSummary):
    file_path: str
    notes: str | None = None


class EvaluateArtifactRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0)
    split: str | None = None
    trace_limit: int | None = Field(default=None, ge=8, le=720)
    capture_render_frames: bool | None = None
    max_render_frames: int | None = Field(default=None, ge=8, le=180)
    render_frame_width: int | None = Field(default=None, ge=480, le=1600)

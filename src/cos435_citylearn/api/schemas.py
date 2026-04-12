from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunnerSummary(BaseModel):
    runner_id: str
    label: str
    algorithm: str
    variant: str
    description: str
    config_path: str
    eval_config_path: str
    launchable: bool


class LaunchJobRequest(BaseModel):
    runner_id: str = "rbc_builtin"
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
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    config_path: str
    eval_config_path: str
    run_id: str | None = None
    average_score: float | None = None
    error_message: str | None = None


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

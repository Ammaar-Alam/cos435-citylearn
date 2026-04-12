export type RunnerSummary = {
  runner_id: string;
  label: string;
  algorithm: string;
  variant: string;
  description: string;
  config_path: string;
  eval_config_path: string;
  launchable: boolean;
  supports_checkpoint_eval: boolean;
};

export type JobSummary = {
  job_id: string;
  runner_id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "orphaned" | "cancelled";
  phase: string | null;
  submitted_at: string;
  started_at: string | null;
  finished_at: string | null;
  pid: number | null;
  config_path: string;
  eval_config_path: string;
  run_id: string | null;
  average_score: number | null;
  error_message: string | null;
  progress_current: number | null;
  progress_total: number | null;
  progress_label: string | null;
  heartbeat_at: string | null;
  latest_preview_path: string | null;
};

export type RunSummary = {
  run_id: string;
  algorithm: string;
  variant: string;
  split: string;
  seed: number;
  dataset_name: string;
  generated_at: string;
  step_count: number;
  average_score: number | null;
  artifacts: Record<string, unknown>;
};

export type RunDetail = {
  summary: RunSummary;
  challenge_metrics: Record<string, { display_name: string; value: number | null; weight: number | null }>;
  district_kpis: Record<string, number | null>;
  manifest: Record<string, unknown>;
};

export type PlaybackTraceFrame = {
  step: number;
  actions: number[][];
  rewards: number[];
  terminated: boolean;
};

export type PlaybackPayload = {
  run_id: string;
  mode: "preview" | "full";
  total_steps: number;
  stored_steps: number;
  truncated: boolean;
  action_names: string[][];
  building_names: string[];
  offset: number;
  limit: number;
  trace_frames: PlaybackTraceFrame[];
  payload: Record<string, any>;
};

export type LaunchJobPayload = {
  runner_id: string;
  artifact_id?: string;
  seed?: number;
  split?: string;
  trace_limit?: number;
  capture_render_frames?: boolean;
  max_render_frames?: number;
  render_frame_width?: number;
};

export type JobState = {
  job_id: string;
  job_kind: string;
  status: string;
  phase: string;
  progress_current: number | null;
  progress_total: number | null;
  progress_label: string | null;
  heartbeat_at: string;
  latest_run_id: string | null;
  latest_preview_path: string | null;
  latest_checkpoint_id: string | null;
  latest_log_offset: number | null;
  error_message: string | null;
};

export type JobEvent = {
  seq: number;
  job_id: string;
  event_type: string;
  created_at: string;
  payload: Record<string, any>;
};

export type JobArtifact = {
  kind: string;
  label: string;
  path: string;
};

export type ArtifactSummary = {
  artifact_id: string;
  artifact_kind: "checkpoint" | "run_bundle" | "simulation_bundle";
  label: string;
  source_filename: string;
  imported_at: string;
  algorithm: string;
  runner_id: string | null;
  status: string;
  evaluable: boolean;
  playback_path: string | null;
  simulation_dir: string | null;
};

export type ArtifactDetail = ArtifactSummary & {
  file_path: string;
  notes: string | null;
};

export type EvaluateArtifactPayload = {
  seed?: number;
  split?: string;
  trace_limit?: number;
  capture_render_frames?: boolean;
  max_render_frames?: number;
  render_frame_width?: number;
};

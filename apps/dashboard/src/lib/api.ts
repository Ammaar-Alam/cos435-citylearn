import type {
  ArtifactDetail,
  ArtifactSummary,
  EvaluateArtifactPayload,
  JobEvent,
  JobArtifact,
  JobState,
  JobSummary,
  LaunchJobPayload,
  PlaybackPayload,
  RunDetail,
  RunSummary,
  RunnerSummary,
} from "../types";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = typeof FormData !== "undefined" && init?.body instanceof FormData;
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function fetchEnvironment(): Promise<Record<string, unknown>> {
  return apiFetch("/api/system/env");
}

export function fetchRunners(): Promise<RunnerSummary[]> {
  return apiFetch("/api/system/runners");
}

export function fetchRuns(): Promise<RunSummary[]> {
  return apiFetch("/api/runs");
}

export function fetchRunDetail(runId: string): Promise<RunDetail> {
  return apiFetch(`/api/runs/${runId}`);
}

export function fetchPlayback(runId: string): Promise<PlaybackPayload> {
  return apiFetch(`/api/runs/${runId}/playback?offset=0&limit=1024`);
}

export function fetchJobs(): Promise<JobSummary[]> {
  return apiFetch("/api/jobs");
}

export function createJob(payload: LaunchJobPayload): Promise<JobSummary> {
  return apiFetch("/api/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function cancelJob(jobId: string): Promise<JobSummary> {
  return apiFetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

export function fetchJobLogs(jobId: string): Promise<{ job_id: string; logs: string }> {
  return apiFetch(`/api/jobs/${jobId}/logs?tail=240`);
}

export function fetchJobState(jobId: string): Promise<JobState> {
  return apiFetch(`/api/jobs/${jobId}/state`);
}

export function fetchJobEvents(jobId: string, afterSeq = 0): Promise<JobEvent[]> {
  return apiFetch(`/api/jobs/${jobId}/events?after_seq=${afterSeq}`);
}

export function fetchJobArtifacts(jobId: string): Promise<JobArtifact[]> {
  return apiFetch(`/api/jobs/${jobId}/artifacts`);
}

export function fetchJobPreview(jobId: string): Promise<PlaybackPayload> {
  return apiFetch(`/api/jobs/${jobId}/preview`);
}

export function fetchArtifacts(): Promise<ArtifactSummary[]> {
  return apiFetch("/api/artifacts");
}

export function fetchArtifact(artifactId: string): Promise<ArtifactDetail> {
  return apiFetch(`/api/artifacts/${artifactId}`);
}

export async function importArtifact(payload: FormData): Promise<ArtifactDetail> {
  return apiFetch("/api/artifacts/import", {
    method: "POST",
    body: payload,
  });
}

export function evaluateArtifact(
  artifactId: string,
  payload: EvaluateArtifactPayload,
): Promise<JobSummary> {
  return apiFetch(`/api/artifacts/${artifactId}/evaluate`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchArtifactPlayback(artifactId: string): Promise<PlaybackPayload> {
  return apiFetch(`/api/artifacts/${artifactId}/playback`);
}

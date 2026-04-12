import type { RunSummary } from "../types";

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return value.toFixed(3);
}

export function formatRelativeTime(value: string): string {
  const date = new Date(value);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function formatMetricLabel(label: string): string {
  return label.replace(/_/g, " ");
}

export function formatDatasetName(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }

  return value.replace(/_/g, " ");
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return `${value.toFixed(1)}%`;
}

export function formatSignedScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(3)}`;
}

export function formatStatusLabel(value: string | null | undefined): string {
  if (!value) {
    return "unknown";
  }

  return value.replace(/_/g, " ");
}

export function formatShortHash(value: string | null | undefined, length = 7): string {
  if (!value) {
    return "—";
  }

  return value.slice(0, length);
}

export function formatRunnerLabel(algorithm: string, variant: string): string {
  return `${algorithm.toUpperCase()} / ${variant.replace(/_/g, " ")}`;
}

export function formatRunTitle(run: Pick<RunSummary, "generated_at" | "seed">): string {
  return `${formatRelativeTime(run.generated_at)} · seed ${run.seed}`;
}

export function formatRunContext(run: Pick<RunSummary, "algorithm" | "variant" | "split">): string {
  return `${formatRunnerLabel(run.algorithm, run.variant)} · ${formatMetricLabel(run.split)}`;
}

export function formatCompactRunId(runId: string): string {
  const stampMatch = runId.match(/__(\d{8}_\d{6})$/);
  if (!stampMatch) {
    return runId;
  }

  const stamp = stampMatch[1];
  return `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)} ${stamp.slice(9, 11)}:${stamp.slice(11, 13)}:${stamp.slice(13, 15)}`;
}

export function getRunArtifactKinds(run: RunSummary): string[] {
  const kinds: string[] = [];

  if (run.artifacts.simulation_export) {
    kinds.push("simulation");
  }
  if (run.artifacts.playback) {
    kinds.push("playback");
  }
  if (run.artifacts.gif) {
    kinds.push("gif");
  }
  if (run.artifacts.poster) {
    kinds.push("poster");
  }

  return kinds;
}

export function getRunArtifactPath(run: RunSummary, key: string): string | null {
  const value = run.artifacts[key];
  return typeof value === "string" ? value : null;
}

export function basename(path: string | null | undefined): string {
  if (!path) {
    return "—";
  }

  return path.split("/").filter(Boolean).pop() ?? path;
}

export function artifactUrl(path: string | null | undefined): string | null {
  if (!path) {
    return null;
  }

  const normalized = path
    .replace(/^\/+/, "")
    .replace(/^results\//, "");

  return `/artifacts/${normalized}`;
}

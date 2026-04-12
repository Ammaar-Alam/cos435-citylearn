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

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return `${value.toFixed(1)}%`;
}

export function artifactUrl(path: string | null | undefined): string | null {
  if (!path) {
    return null;
  }

  return `/artifacts/${path}`;
}

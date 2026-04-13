from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cos435_citylearn.api.schemas import RunDetail, RunSummary
from cos435_citylearn.api.settings import ApiSettings


class RunStore:
    def __init__(self, settings: ApiSettings):
        self.settings = settings

    def _run_dir(self, run_id: str) -> Path:
        return self.settings.run_root / run_id

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    def _normalize_repo_path(self, value: str | None) -> str | None:
        if not value:
            return value

        path = Path(value)
        if not path.is_absolute():
            return value

        for root in (self.settings.artifacts_root, self.settings.repo_root):
            try:
                return str(path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                continue

        return value

    def _summary_from_paths(self, manifest_path: Path, metrics_path: Path) -> RunSummary:
        manifest = self._load_json(manifest_path)
        metrics = self._load_json(metrics_path)
        trace_path = manifest_path.parent / "rollout_trace.json"
        playback_manifest_path = manifest_path.parent / "playback_manifest.json"
        simulation_dir = self._normalize_repo_path(manifest.get("simulation_dir"))
        playback_payload = (
            self._load_json(playback_manifest_path) if playback_manifest_path.exists() else {}
        )
        media = playback_payload.get("media", {})
        trace_steps = (
            int(playback_payload.get("decision_steps", manifest["step_count"]))
            if trace_path.exists()
            else 0
        )

        artifacts = {
            "metrics": True,
            "rollout_trace": trace_path.exists(),
            "trace_steps": trace_steps,
            "playback": bool(manifest.get("playback_path")),
            "simulation_export": bool(simulation_dir),
            "gif": bool(media.get("gif_path")),
            "poster": self._normalize_repo_path(media.get("poster_path")),
            "simulation_dir": simulation_dir,
        }

        return RunSummary(
            run_id=manifest["run_id"],
            algorithm=metrics["algorithm"],
            variant=metrics["variant"],
            split=metrics["split"],
            seed=metrics["seed"],
            dataset_name=metrics["dataset_name"],
            generated_at=manifest["generated_at"],
            step_count=int(manifest["step_count"]),
            average_score=metrics["average_score"],
            artifacts=artifacts,
        )

    def list_runs(self) -> list[RunSummary]:
        runs = []

        for manifest_path in self.settings.run_root.glob("*/manifest.json"):
            metrics_path = manifest_path.parent / "metrics.json"
            if not metrics_path.exists():
                continue
            runs.append(self._summary_from_paths(manifest_path, metrics_path))

        runs.sort(key=lambda item: item.generated_at, reverse=True)
        return runs

    def get_run(self, run_id: str) -> RunDetail:
        run_dir = self._run_dir(run_id)
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        if not manifest_path.exists() or not metrics_path.exists():
            raise KeyError(f"unknown run: {run_id}")

        summary = self._summary_from_paths(manifest_path, metrics_path)
        metrics = self._load_json(metrics_path)
        manifest = self._load_json(manifest_path)

        return RunDetail(
            summary=summary,
            challenge_metrics=metrics["challenge_metrics"],
            district_kpis=metrics["district_kpis"],
            manifest=manifest,
        )

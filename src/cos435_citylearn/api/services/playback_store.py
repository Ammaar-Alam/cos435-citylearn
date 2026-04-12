from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cos435_citylearn.api.schemas import PlaybackFrame, PlaybackResponse
from cos435_citylearn.api.settings import ApiSettings


class PlaybackStore:
    def __init__(self, settings: ApiSettings):
        self.settings = settings

    def _resolve_repo_path(self, artifact_path: str) -> Path:
        candidate = Path(artifact_path)
        if candidate.is_absolute():
            return candidate

        repo_path = self.settings.repo_root / candidate
        if repo_path.exists():
            return repo_path

        results_path = self.settings.artifacts_root / candidate
        return results_path

    def _load_json(self, path: Path) -> dict[str, Any] | list[Any]:
        return json.loads(path.read_text())

    def _response_from_payload(
        self,
        *,
        run_id: str,
        mode: str,
        payload: dict[str, Any],
        offset: int,
        limit: int,
    ) -> PlaybackResponse:
        trace = payload.get("trace", [])
        trace_slice = trace[offset : offset + limit]
        total_steps = int(
            payload.get("decision_steps", payload.get("episode_total_steps", len(trace)))
        )
        return PlaybackResponse(
            run_id=run_id,
            mode=mode,
            total_steps=total_steps,
            stored_steps=len(trace),
            truncated=len(trace) < total_steps,
            action_names=payload.get("action_names", []),
            building_names=payload.get("building_names", []),
            offset=offset,
            limit=limit,
            trace_frames=[PlaybackFrame(**frame) for frame in trace_slice],
            payload=payload,
        )

    def get_playback(self, run_id: str, *, offset: int = 0, limit: int = 256) -> PlaybackResponse:
        playback_path = self.settings.ui_exports_root / "playback" / f"{run_id}.json"
        if playback_path.exists():
            payload = self._load_json(playback_path)
            return self._response_from_payload(
                run_id=run_id,
                mode="full",
                payload=payload,
                offset=offset,
                limit=limit,
            )

        run_dir = self.settings.run_root / run_id
        trace_path = run_dir / "rollout_trace.json"
        if not trace_path.exists():
            raise KeyError(f"no playback available for run: {run_id}")

        schema_path = self.settings.manifests_root / "observation_action_schema.json"
        schema = self._load_json(schema_path) if schema_path.exists() else {}
        trace = self._load_json(trace_path)
        trace_slice = trace[offset : offset + limit]
        step_count = int(self._load_json(run_dir / "manifest.json")["step_count"])

        return PlaybackResponse(
            run_id=run_id,
            mode="preview",
            total_steps=step_count,
            stored_steps=len(trace),
            truncated=len(trace) < step_count,
            action_names=schema.get("action_names", []),
            building_names=schema.get("building_names", []),
            offset=offset,
            limit=limit,
            trace_frames=[PlaybackFrame(**frame) for frame in trace_slice],
            payload={},
        )

    def get_job_preview(self, job_id: str) -> PlaybackResponse:
        preview_path = self.settings.jobs_root / job_id / "preview.json"
        if not preview_path.exists():
            raise KeyError(f"no live preview available for job: {job_id}")

        payload = self._load_json(preview_path)
        return self._response_from_payload(
            run_id=payload.get("run_id") or job_id,
            mode="preview",
            offset=0,
            limit=len(payload.get("trace", [])),
            payload=payload,
        )

    def get_artifact_playback(self, artifact_path: str) -> PlaybackResponse:
        path = self._resolve_repo_path(artifact_path)
        if not path.exists():
            raise KeyError(f"artifact playback not found: {artifact_path}")

        payload = self._load_json(path)
        run_id = str(payload.get("run_id", path.stem))
        return self._response_from_payload(
            run_id=run_id,
            mode="full",
            offset=0,
            limit=len(payload.get("trace", [])),
            payload=payload,
        )

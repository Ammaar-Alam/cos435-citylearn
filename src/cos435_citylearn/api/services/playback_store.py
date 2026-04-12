from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cos435_citylearn.api.schemas import PlaybackFrame, PlaybackResponse
from cos435_citylearn.api.settings import ApiSettings


class PlaybackStore:
    def __init__(self, settings: ApiSettings):
        self.settings = settings

    def _load_json(self, path: Path) -> dict[str, Any] | list[Any]:
        return json.loads(path.read_text())

    def get_playback(self, run_id: str, *, offset: int = 0, limit: int = 256) -> PlaybackResponse:
        playback_path = self.settings.ui_exports_root / "playback" / f"{run_id}.json"
        if playback_path.exists():
            payload = self._load_json(playback_path)
            trace = payload.get("trace", [])
            trace_slice = trace[offset : offset + limit]
            total_steps = int(payload.get("decision_steps", len(trace)))
            return PlaybackResponse(
                run_id=run_id,
                mode="full",
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

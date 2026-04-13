from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cos435_citylearn.api.services.job_event_store import JobEventStore
from cos435_citylearn.api.services.job_state_store import JobStateStore
from cos435_citylearn.io import write_json_atomic
from cos435_citylearn.runtime import utc_now_iso


@dataclass
class WorkloadContext:
    job_id: str
    job_dir: Path
    state_store: JobStateStore
    event_store: JobEventStore
    job_kind: str

    def start(self, *, phase: str, total: int | None = None, label: str | None = None) -> None:
        payload = {
            "job_id": self.job_id,
            "job_kind": self.job_kind,
            "status": "running",
            "phase": phase,
            "progress_current": 0,
            "progress_total": total,
            "progress_label": label,
            "heartbeat_at": utc_now_iso(),
            "latest_run_id": None,
            "latest_preview_path": None,
            "latest_checkpoint_id": None,
            "latest_log_offset": None,
            "error_message": None,
        }
        self.state_store.write(self.job_id, payload)
        self.event_store.append(
            self.job_id,
            {
                "job_id": self.job_id,
                "event_type": "job_started",
                "created_at": utc_now_iso(),
                "payload": {"phase": phase, "progress_total": total, "progress_label": label},
            },
        )

    def update(
        self,
        *,
        phase: str,
        current: int,
        total: int | None,
        label: str | None = None,
        preview_payload: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        state = self.state_store.get(self.job_id) or {
            "job_id": self.job_id,
            "job_kind": self.job_kind,
        }
        preview_path = None
        if preview_payload is not None:
            preview_path = self.job_dir / "preview.json"
            write_json_atomic(preview_path, preview_payload)

        next_state = {
            **state,
            "status": "running",
            "phase": phase,
            "progress_current": current,
            "progress_total": total,
            "progress_label": label,
            "heartbeat_at": utc_now_iso(),
            "latest_run_id": run_id or state.get("latest_run_id"),
            "latest_preview_path": (
                str(preview_path)
                if preview_path is not None
                else state.get("latest_preview_path")
            ),
            "latest_checkpoint_id": state.get("latest_checkpoint_id"),
            "latest_log_offset": state.get("latest_log_offset"),
            "error_message": None,
        }
        self.state_store.write(self.job_id, next_state)
        self.event_store.append(
            self.job_id,
            {
                "job_id": self.job_id,
                "event_type": "progress",
                "created_at": utc_now_iso(),
                "payload": {
                    "phase": phase,
                    "progress_current": current,
                    "progress_total": total,
                    "progress_label": label,
                    "preview_path": str(preview_path) if preview_path is not None else None,
                    "run_id": run_id,
                },
            },
        )

    def artifact(self, *, kind: str, path: str, label: str) -> None:
        artifacts_path = self.job_dir / "artifacts.json"
        existing = []
        if artifacts_path.exists():
            import json

            existing = json.loads(artifacts_path.read_text())
        existing.append({"kind": kind, "path": path, "label": label})
        write_json_atomic(artifacts_path, existing)
        self.event_store.append(
            self.job_id,
            {
                "job_id": self.job_id,
                "event_type": "artifact_written",
                "created_at": utc_now_iso(),
                "payload": {"kind": kind, "path": path, "label": label},
            },
        )

    def finish(self, *, result: dict[str, Any]) -> None:
        state = self.state_store.get(self.job_id) or {
            "job_id": self.job_id,
            "job_kind": self.job_kind,
        }
        next_state = {
            **state,
            "status": "succeeded",
            "phase": "completed",
            "heartbeat_at": utc_now_iso(),
            "latest_run_id": result.get("run_id", state.get("latest_run_id")),
            "error_message": None,
        }
        self.state_store.write(self.job_id, next_state)
        self.event_store.append(
            self.job_id,
            {
                "job_id": self.job_id,
                "event_type": "job_finished",
                "created_at": utc_now_iso(),
                "payload": {
                    "run_id": result.get("run_id"),
                    "average_score": result.get("average_score"),
                },
            },
        )

    def fail(self, *, error_message: str) -> None:
        state = self.state_store.get(self.job_id) or {
            "job_id": self.job_id,
            "job_kind": self.job_kind,
        }
        next_state = {
            **state,
            "status": "failed",
            "phase": "failed",
            "heartbeat_at": utc_now_iso(),
            "error_message": error_message,
        }
        self.state_store.write(self.job_id, next_state)
        self.event_store.append(
            self.job_id,
            {
                "job_id": self.job_id,
                "event_type": "job_failed",
                "created_at": utc_now_iso(),
                "payload": {"error_message": error_message},
            },
        )

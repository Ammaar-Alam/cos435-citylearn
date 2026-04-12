from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from cos435_citylearn.api.schemas import JobSummary, LaunchJobRequest
from cos435_citylearn.api.services.job_event_store import JobEventStore
from cos435_citylearn.api.services.job_state_store import JobStateStore
from cos435_citylearn.api.services.runner_registry import get_runner, materialize_runner_files
from cos435_citylearn.api.settings import ApiSettings
from cos435_citylearn.config import load_yaml
from cos435_citylearn.env.loader import resolve_schema_path
from cos435_citylearn.io import write_json
from cos435_citylearn.runtime import utc_now_iso


class JobManager:
    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._log_handles: dict[str, Any] = {}
        self._queue: deque[str] = deque()
        self.settings.jobs_root.mkdir(parents=True, exist_ok=True)
        self.state_store = JobStateStore(self.settings.jobs_root)
        self.event_store = JobEventStore(self.settings.jobs_root)
        self._recover_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_root / job_id

    def _job_state_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _job_request_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "request.json"

    def _job_result_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "result.json"

    def _job_error_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "error.json"

    def _job_log_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "process.log"

    def _job_state_file_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "state.json"

    def _job_preview_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "preview.json"

    def _job_artifacts_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "artifacts.json"

    def _load_job(self, job_id: str) -> dict[str, Any]:
        path = self._job_state_path(job_id)
        if not path.exists():
            raise KeyError(f"unknown job: {job_id}")
        return json.loads(path.read_text())

    def _write_job(self, job_id: str, payload: dict[str, Any]) -> None:
        write_json(self._job_state_path(job_id), payload)

    def _recover_jobs(self) -> None:
        for job_file in sorted(self.settings.jobs_root.glob("*/job.json")):
            payload = json.loads(job_file.read_text())
            if payload["status"] in {"running", "queued"}:
                payload["status"] = "orphaned"
                payload["phase"] = "orphaned"
                payload["finished_at"] = utc_now_iso()
                payload["error_message"] = "backend restarted before the job finished"
                self._write_job(payload["job_id"], payload)
                previous_state = self.state_store.get(payload["job_id"]) or {}
                self.state_store.write(
                    payload["job_id"],
                    {
                        **previous_state,
                        "job_id": payload["job_id"],
                        "job_kind": previous_state.get("job_kind", "evaluation"),
                        "status": "orphaned",
                        "phase": "orphaned",
                        "progress_current": None,
                        "progress_total": None,
                        "progress_label": "backend restarted before the job finished",
                        "heartbeat_at": utc_now_iso(),
                        "latest_run_id": previous_state.get("latest_run_id") or payload.get("run_id"),
                        "latest_preview_path": None,
                        "latest_checkpoint_id": previous_state.get("latest_checkpoint_id"),
                        "latest_log_offset": previous_state.get("latest_log_offset"),
                        "error_message": "backend restarted before the job finished",
                    },
                )

    def _running_job_count(self) -> int:
        return sum(1 for job_id in self._processes if self._processes[job_id].poll() is None)

    def submit(self, request: LaunchJobRequest) -> JobSummary:
        spec = get_runner(request.runner_id)
        if not spec.launchable:
            raise ValueError(f"{request.runner_id} is not launchable yet")

        with self._lock:
            self.refresh()
            timestamp = utc_now_iso().replace(":", "").replace("-", "").replace("+00:00", "z")
            job_id = f"job_{timestamp}_{uuid4().hex[:8]}"
            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            config_path, eval_config_path = materialize_runner_files(request, job_dir=job_dir)
            config = load_yaml(config_path)
            split_path = f"configs/splits/{config['env']['split']}.yaml"
            resolve_schema_path(config["env"]["base_config"], split_path)

            request_payload = {
                "job_id": job_id,
                "runner_id": request.runner_id,
                "artifact_id": request.artifact_id,
                "job_kind": "evaluation",
                "workload_id": spec.workload_id,
                "config_path": str(config_path),
                "eval_config_path": str(eval_config_path),
                "job_dir": str(job_dir),
                "result_path": str(self._job_result_path(job_id)),
                "error_path": str(self._job_error_path(job_id)),
                "output_root": str(self.settings.run_root),
                "metrics_root": str(self.settings.results_root / "metrics"),
                "manifests_root": str(self.settings.manifests_root),
                "ui_exports_root": str(self.settings.ui_exports_root),
                "artifacts_root": str(self.settings.artifacts_root),
            }
            write_json(self._job_request_path(job_id), request_payload)

            state = {
                "job_id": job_id,
                "runner_id": request.runner_id,
                "status": "queued",
                "submitted_at": utc_now_iso(),
                "started_at": None,
                "finished_at": None,
                "pid": None,
                "config_path": str(config_path),
                "eval_config_path": str(eval_config_path),
                "run_id": None,
                "average_score": None,
                "error_message": None,
                "phase": "queued",
                "progress_current": None,
                "progress_total": None,
                "progress_label": None,
                "heartbeat_at": None,
                "latest_preview_path": None,
            }
            self._write_job(job_id, state)
            self.state_store.write(
                job_id,
                {
                    "job_id": job_id,
                    "job_kind": "evaluation",
                    "status": "queued",
                    "phase": "queued",
                    "progress_current": 0,
                    "progress_total": None,
                    "progress_label": "queued",
                    "heartbeat_at": utc_now_iso(),
                    "latest_run_id": None,
                    "latest_preview_path": None,
                    "latest_checkpoint_id": None,
                    "latest_log_offset": None,
                    "error_message": None,
                },
            )
            self.event_store.append(
                job_id,
                {
                    "job_id": job_id,
                    "event_type": "job_submitted",
                    "created_at": utc_now_iso(),
                    "payload": {"runner_id": request.runner_id},
                },
            )
            self._queue.append(job_id)
            self._drain_queue()
            return JobSummary(**self._load_job(job_id))

    def _drain_queue(self) -> None:
        while self._queue and self._running_job_count() < self.settings.max_concurrent_jobs:
            job_id = self._queue.popleft()
            state = self._load_job(job_id)
            if state["status"] != "queued":
                continue
            self._start(job_id, state)

    def _start(self, job_id: str, state: dict[str, Any]) -> None:
        request_path = self._job_request_path(job_id)
        log_handle = self._job_log_path(job_id).open("a", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["COS435_REQUIRE_DATA"] = "1"
        env["MPLCONFIGDIR"] = str(self.settings.mpl_config_dir)

        process = subprocess.Popen(
            [
                str(self.settings.python_executable),
                "-m",
                "cos435_citylearn.api.worker_main",
                "--job-file",
                str(request_path),
            ],
            cwd=self.settings.repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._processes[job_id] = process
        self._log_handles[job_id] = log_handle
        state["status"] = "running"
        state["started_at"] = utc_now_iso()
        state["pid"] = process.pid
        self._write_job(job_id, state)
        self.event_store.append(
            job_id,
            {
                "job_id": job_id,
                "event_type": "process_spawned",
                "created_at": utc_now_iso(),
                "payload": {"pid": process.pid},
            },
        )

    def _merge_live_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.state_store.get(payload["job_id"])
        if state is None:
            return payload

        merged = {**payload}
        for key in [
            "phase",
            "progress_current",
            "progress_total",
            "progress_label",
            "heartbeat_at",
        ]:
            merged[key] = state.get(key)
        merged["latest_preview_path"] = state.get("latest_preview_path")
        merged["error_message"] = state.get("error_message") or payload.get("error_message")
        return merged

    def refresh(self) -> None:
        finished = []

        for job_id, process in list(self._processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            finished.append((job_id, return_code))

        for job_id, return_code in finished:
            state = self._load_job(job_id)
            state["finished_at"] = utc_now_iso()
            log_handle = self._log_handles.pop(job_id, None)
            if log_handle is not None:
                log_handle.close()

            self._processes.pop(job_id, None)

            if return_code == 0 and self._job_result_path(job_id).exists():
                result = json.loads(self._job_result_path(job_id).read_text())
                state["status"] = "succeeded"
                state["run_id"] = result.get("run_id")
                state["average_score"] = result.get("average_score")
            elif state["status"] == "cancelled":
                pass
            else:
                state["status"] = "failed"
                if self._job_error_path(job_id).exists():
                    error_payload = json.loads(self._job_error_path(job_id).read_text())
                    state["error_message"] = error_payload.get("error")

            self._write_job(job_id, state)

        self._drain_queue()

    def list_jobs(self) -> list[JobSummary]:
        with self._lock:
            self.refresh()
            jobs = [
                json.loads(path.read_text())
                for path in self.settings.jobs_root.glob("*/job.json")
            ]
            jobs = [self._merge_live_state(job) for job in jobs]
            jobs.sort(key=lambda item: item["submitted_at"], reverse=True)
            return [JobSummary(**job) for job in jobs]

    def get_job(self, job_id: str) -> JobSummary:
        with self._lock:
            self.refresh()
            return JobSummary(**self._merge_live_state(self._load_job(job_id)))

    def cancel(self, job_id: str) -> JobSummary:
        with self._lock:
            self.refresh()
            state = self._load_job(job_id)
            process = self._processes.get(job_id)
            cancelled = False
            if process is not None and process.poll() is None:
                process.terminate()
                cancelled = True
            elif state["status"] == "queued":
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                cancelled = True
            if not cancelled:
                return JobSummary(**self._merge_live_state(state))

            state["status"] = "cancelled"
            state["phase"] = "cancelled"
            state["finished_at"] = utc_now_iso()
            self._write_job(job_id, state)
            self.state_store.write(
                job_id,
                {
                    **(self.state_store.get(job_id) or {}),
                    "job_id": job_id,
                    "job_kind": "evaluation",
                    "status": "cancelled",
                    "phase": "cancelled",
                    "heartbeat_at": utc_now_iso(),
                    "error_message": None,
                },
            )
            return JobSummary(**self._merge_live_state(state))

    def tail_logs(self, job_id: str, tail: int = 200) -> str:
        if not self._job_dir(job_id).exists():
            raise KeyError(f"unknown job: {job_id}")
        log_path = self._job_log_path(job_id)
        if not log_path.exists():
            return ""
        return "\n".join(log_path.read_text().splitlines()[-tail:])

    def get_state(self, job_id: str) -> dict[str, Any]:
        state = self.state_store.get(job_id)
        if state is not None:
            return state

        job = self._load_job(job_id)
        return {
            "job_id": job_id,
            "job_kind": "evaluation",
            "status": job["status"],
            "phase": job.get("phase") or job["status"],
            "progress_current": job.get("progress_current"),
            "progress_total": job.get("progress_total"),
            "progress_label": job.get("progress_label"),
            "heartbeat_at": (
                job.get("heartbeat_at")
                or job.get("finished_at")
                or job["submitted_at"]
            ),
            "latest_run_id": job.get("run_id"),
            "latest_preview_path": job.get("latest_preview_path"),
            "latest_checkpoint_id": None,
            "latest_log_offset": None,
            "error_message": job.get("error_message"),
        }

    def get_events(self, job_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        if not self._job_dir(job_id).exists():
            raise KeyError(f"unknown job: {job_id}")
        return self.event_store.list_after(job_id, after_seq=after_seq)

    def get_preview_path(self, job_id: str) -> Path:
        preview_path = self._job_preview_path(job_id)
        if not preview_path.exists():
            raise KeyError(f"no preview available for job: {job_id}")
        return preview_path

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        if not self._job_dir(job_id).exists():
            raise KeyError(f"unknown job: {job_id}")
        path = self._job_artifacts_path(job_id)
        if not path.exists():
            return []
        return json.loads(path.read_text())

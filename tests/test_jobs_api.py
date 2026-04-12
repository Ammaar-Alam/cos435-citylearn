from __future__ import annotations

import threading
from pathlib import Path

from fastapi.testclient import TestClient

from cos435_citylearn.api.app import create_app
from cos435_citylearn.api.schemas import JobSummary
from cos435_citylearn.api.services.job_manager import JobManager
from cos435_citylearn.api.settings import ApiSettings
from cos435_citylearn.io import write_json
from cos435_citylearn.paths import CONFIGS_DIR, REPO_ROOT


def build_test_settings(tmp_path: Path) -> ApiSettings:
    frontend_root = tmp_path / "dashboard"
    frontend_dist = frontend_root / "dist"
    frontend_dist.mkdir(parents=True, exist_ok=True)
    (frontend_dist / "index.html").write_text("<html><body>dashboard</body></html>")

    return ApiSettings(
        repo_root=tmp_path,
        config_root=CONFIGS_DIR,
        results_root=tmp_path / "results",
        run_root=tmp_path / "results" / "runs",
        manifests_root=tmp_path / "results" / "manifests",
        ui_exports_root=tmp_path / "results" / "ui_exports",
        jobs_root=tmp_path / "results" / "dashboard" / "jobs",
        imported_artifacts_root=tmp_path / "results" / "dashboard" / "artifacts",
        artifacts_root=tmp_path / "results",
        frontend_root=frontend_root,
        frontend_dist=frontend_dist,
        python_executable=REPO_ROOT / ".venv" / "bin" / "python",
        mpl_config_dir=tmp_path / ".cache" / "matplotlib",
        max_concurrent_jobs=1,
    )


def test_cancel_keeps_finished_job_state(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = JobManager(settings)
    job_id = "job_finished"
    job_dir = settings.jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        job_dir / "job.json",
        {
            "job_id": job_id,
            "runner_id": "rbc_builtin",
            "status": "succeeded",
            "submitted_at": "2026-04-12T00:00:00Z",
            "started_at": "2026-04-12T00:00:10Z",
            "finished_at": "2026-04-12T00:01:00Z",
            "pid": None,
            "config_path": "config.yaml",
            "eval_config_path": "eval.yaml",
            "run_id": "run_1",
            "average_score": 1.23,
            "error_message": None,
            "phase": "completed",
            "progress_current": 10,
            "progress_total": 10,
            "progress_label": "done",
            "heartbeat_at": "2026-04-12T00:01:00Z",
            "latest_preview_path": None,
        },
    )
    manager.state_store.write(
        job_id,
        {
            "job_id": job_id,
            "job_kind": "evaluation",
            "status": "succeeded",
            "phase": "completed",
            "progress_current": 10,
            "progress_total": 10,
            "progress_label": "done",
            "heartbeat_at": "2026-04-12T00:01:00Z",
            "latest_run_id": "run_1",
            "latest_preview_path": None,
            "latest_checkpoint_id": None,
            "latest_log_offset": None,
            "error_message": None,
        },
    )

    result = manager.cancel(job_id)

    assert result.status == "succeeded"
    assert result.phase == "completed"
    assert manager.state_store.get(job_id)["status"] == "succeeded"
    assert manager.state_store.get(job_id)["phase"] == "completed"


def test_unknown_job_logs_and_artifacts_return_404(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    logs_response = client.get("/api/jobs/job_missing/logs")
    artifacts_response = client.get("/api/jobs/job_missing/artifacts")

    assert logs_response.status_code == 404
    assert artifacts_response.status_code == 404


def test_create_job_maps_missing_split_file_to_400(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    app = create_app(settings)

    def raise_missing_file(_payload) -> JobSummary:
        raise FileNotFoundError("unknown split config")

    app.state.job_manager.submit = raise_missing_file  # type: ignore[method-assign]
    client = TestClient(app)

    response = client.post("/api/jobs", json={"runner_id": "rbc_builtin", "split": "missing"})

    assert response.status_code == 400
    assert response.json()["detail"] == "unknown split config"


def test_recover_jobs_marks_state_store_orphaned(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    job_id = "job_orphaned"
    job_dir = settings.jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        job_dir / "job.json",
        {
            "job_id": job_id,
            "runner_id": "rbc_builtin",
            "status": "running",
            "submitted_at": "2026-04-12T00:00:00Z",
            "started_at": "2026-04-12T00:00:10Z",
            "finished_at": None,
            "pid": 12345,
            "config_path": "config.yaml",
            "eval_config_path": "eval.yaml",
            "run_id": "run_1",
            "average_score": None,
            "error_message": None,
            "phase": "rollout",
            "progress_current": 3,
            "progress_total": 10,
            "progress_label": "rollout",
            "heartbeat_at": "2026-04-12T00:00:20Z",
            "latest_preview_path": None,
        },
    )
    write_json(
        job_dir / "state.json",
        {
            "job_id": job_id,
            "job_kind": "evaluation",
            "status": "running",
            "phase": "rollout",
            "progress_current": 3,
            "progress_total": 10,
            "progress_label": "rollout",
            "heartbeat_at": "2026-04-12T00:00:20Z",
            "latest_run_id": "run_1",
            "latest_preview_path": "preview.json",
            "latest_checkpoint_id": None,
            "latest_log_offset": 42,
            "error_message": None,
        },
    )

    manager = JobManager(settings)
    job = manager.get_job(job_id)
    state = manager.get_state(job_id)

    assert job.status == "orphaned"
    assert job.phase == "orphaned"
    assert state["status"] == "orphaned"
    assert state["phase"] == "orphaned"
    assert state["latest_preview_path"] is None
    assert state["progress_current"] is None
    assert state["progress_total"] is None


def test_watch_process_triggers_refresh_when_subprocess_exits(tmp_path: Path) -> None:
    class DummyProcess:
        def __init__(self) -> None:
            self._done = threading.Event()

        def poll(self):
            return 0 if self._done.is_set() else None

        def wait(self):
            self._done.wait(timeout=2)
            return 0

        def finish(self) -> None:
            self._done.set()

    settings = build_test_settings(tmp_path)
    manager = JobManager(settings)
    process = DummyProcess()
    manager._processes["job_watch"] = process  # type: ignore[assignment]

    refreshed = threading.Event()

    def fake_refresh() -> None:
        refreshed.set()

    manager.refresh = fake_refresh  # type: ignore[method-assign]

    watcher = threading.Thread(target=manager._watch_process, args=("job_watch", process), daemon=True)
    watcher.start()
    process.finish()
    watcher.join(timeout=2)

    assert refreshed.wait(timeout=2)

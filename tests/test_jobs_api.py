from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cos435_citylearn.api.app import create_app
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

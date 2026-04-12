from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from cos435_citylearn.api.app import create_app
from cos435_citylearn.api.settings import ApiSettings
from cos435_citylearn.paths import CONFIGS_DIR, REPO_ROOT


def build_test_settings(tmp_path: Path) -> ApiSettings:
    frontend_root = REPO_ROOT / "apps" / "dashboard"
    return ApiSettings(
        repo_root=REPO_ROOT,
        config_root=CONFIGS_DIR,
        results_root=tmp_path,
        run_root=tmp_path / "runs",
        manifests_root=tmp_path / "manifests",
        ui_exports_root=tmp_path / "ui_exports",
        jobs_root=tmp_path / "dashboard" / "jobs",
        imported_artifacts_root=tmp_path / "dashboard" / "artifacts",
        artifacts_root=tmp_path,
        frontend_root=frontend_root,
        frontend_dist=frontend_root / "dist",
        python_executable=REPO_ROOT / ".venv" / "bin" / "python",
        mpl_config_dir=tmp_path / ".cache" / "matplotlib",
        max_concurrent_jobs=1,
    )


def test_job_api_emits_preview_and_progress_events(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/jobs",
        json={
            "runner_id": "rbc_builtin",
            "trace_limit": 64,
            "capture_render_frames": False,
            "max_render_frames": 8,
            "render_frame_width": 720,
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    preview_seen = False
    progress_seen = False

    deadline = time.time() + 45
    final_status = None
    run_id = None
    while time.time() < deadline:
        job_response = client.get(f"/api/jobs/{job_id}")
        assert job_response.status_code == 200, job_response.text
        job_payload = job_response.json()
        final_status = job_payload["status"]
        run_id = job_payload.get("run_id")

        events_response = client.get(f"/api/jobs/{job_id}/events")
        assert events_response.status_code == 200, events_response.text
        events = events_response.json()
        progress_seen = progress_seen or any(event["event_type"] == "progress" for event in events)

        preview_response = client.get(f"/api/jobs/{job_id}/preview")
        if preview_response.status_code == 200:
            preview_seen = True

        if final_status in {"succeeded", "failed", "cancelled"}:
            break

        time.sleep(0.5)

    assert final_status == "succeeded"
    assert preview_seen
    assert progress_seen
    assert run_id

    runs_response = client.get("/api/runs")
    assert runs_response.status_code == 200, runs_response.text
    assert any(run["run_id"] == run_id for run in runs_response.json())

    playback_response = client.get(f"/api/runs/{run_id}/playback?offset=0&limit=50000")
    assert playback_response.status_code == 200, playback_response.text
    playback_payload = playback_response.json()
    assert playback_payload["mode"] == "full"
    assert playback_payload["stored_steps"] >= playback_payload["total_steps"] - 1

    assert (tmp_path / "runs" / run_id / "manifest.json").exists()
    assert (tmp_path / "ui_exports" / "playback" / f"{run_id}.json").exists()

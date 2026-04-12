from __future__ import annotations

import json
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
        python_executable=Path(".venv/bin/python"),
        mpl_config_dir=tmp_path / ".cache" / "matplotlib",
        max_concurrent_jobs=1,
    )


def test_artifact_import_round_trips_playback_payload(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))
    playback_payload = {
        "run_id": "imported_run",
        "episode_total_steps": 4,
        "decision_steps": 3,
        "action_names": [["battery"]],
        "building_names": ["Building_1"],
        "trace": [
            {"step": 0, "actions": [[0.1]], "rewards": [0.0], "terminated": False},
            {"step": 1, "actions": [[0.2]], "rewards": [0.1], "terminated": False},
            {"step": 2, "actions": [[0.0]], "rewards": [0.2], "terminated": True},
        ],
        "buildings": [{"name": "Building_1", "series": {"battery_soc": [12.0, 18.0, 21.0]}}],
        "district": {"net_electricity_consumption": [1.0, 0.8, 0.6]},
        "timestamps": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", "2023-01-01T02:00:00Z"],
        "ui_export": {"simulation_dir": None},
        "media": {
            "poster_path": None,
            "gif_path": None,
            "frames": [],
            "frame_count": 0,
            "frame_stride": None,
        },
        "metrics": {},
        "algorithm": "sac",
        "variant": "checkpoint_eval",
        "split": "public_dev",
        "seed": 0,
        "dataset_name": "imported",
    }
    payload_bytes = json.dumps(playback_payload).encode("utf-8")

    response = client.post(
        "/api/artifacts/import",
        data={
            "artifact_kind": "run_bundle",
            "label": "remote playback",
            "notes": "imported from remote training",
        },
        files={"file": ("playback.json", payload_bytes, "application/json")},
    )
    assert response.status_code == 200, response.text
    artifact = response.json()
    assert artifact["status"] == "inspectable"
    assert artifact["playback_path"].endswith("playback.json")

    playback_response = client.get(f"/api/artifacts/{artifact['artifact_id']}/playback")
    assert playback_response.status_code == 200, playback_response.text
    playback = playback_response.json()
    assert playback["run_id"] == "imported_run"
    assert playback["stored_steps"] == 3
    assert playback["total_steps"] == 4
    assert playback["mode"] == "full"


def test_artifact_import_rejects_unknown_artifact_kind(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/artifacts/import",
        data={
            "artifact_kind": "mystery_blob",
            "label": "bad kind",
        },
        files={"file": ("artifact.bin", b"payload", "application/octet-stream")},
    )

    assert response.status_code == 422

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from cos435_citylearn.api.app import create_app
from cos435_citylearn.api.schemas import JobSummary
from cos435_citylearn.api.settings import ApiSettings
from cos435_citylearn.io import write_json
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
    assert artifact["playback_path"] == artifact["file_path"]
    assert not artifact["file_path"].startswith("/")

    playback_response = client.get(f"/api/artifacts/{artifact['artifact_id']}/playback")
    assert playback_response.status_code == 200, playback_response.text
    playback = playback_response.json()
    assert playback["run_id"] == "imported_run"
    assert playback["stored_steps"] == 3
    assert playback["total_steps"] == 3
    assert playback["truncated"] is False
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


def test_artifact_evaluate_maps_missing_split_file_to_400(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    app = create_app(settings)

    def build_request(_artifact_id, _payload):
        return {"runner_id": "rbc_builtin"}

    def raise_missing_file(_payload) -> JobSummary:
        raise FileNotFoundError("unknown split config")

    app.state.artifact_store.build_evaluation_request = build_request  # type: ignore[method-assign]
    app.state.job_manager.submit = raise_missing_file  # type: ignore[method-assign]
    client = TestClient(app)

    response = client.post("/api/artifacts/artifact_123/evaluate", json={"split": "missing"})

    assert response.status_code == 400
    assert response.json()["detail"] == "unknown split config"


def test_artifact_evaluate_rejects_incompatible_sac_checkpoint_before_queueing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = build_test_settings(tmp_path)
    app = create_app(settings)
    artifact_id = "artifact_bad_sac"
    artifact_dir = settings.imported_artifacts_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        artifact_dir / "artifact.json",
        {
            "artifact_id": artifact_id,
            "artifact_kind": "checkpoint",
            "label": "bad checkpoint",
            "source_filename": "checkpoint.pt",
            "imported_at": "2026-04-13T00:00:00Z",
            "algorithm": "sac",
            "runner_id": "sac_central_baseline",
            "status": "evaluable",
            "file_path": "dashboard/artifacts/artifact_bad_sac/checkpoint.pt",
            "notes": None,
            "evaluable": True,
            "playback_path": None,
            "simulation_dir": None,
        },
    )

    def fake_resolve_imported_checkpoint_path(**_kwargs):
        return tmp_path / "checkpoint.pt"

    def fake_safe_load_checkpoint_payload(_path):
        return {
            "algorithm": "sac",
            "control_mode": "shared_dtde",
            "observation_names": [["hour", "load"]],
            "action_names": [["battery"]],
            "controller_state": {
                "controller_type": "shared_parameter_sac",
                "hidden_dimension": [64, 64],
                "discount": 0.99,
                "tau": 0.005,
                "alpha": 0.2,
                "lr": 0.0003,
                "batch_size": 16,
                "replay_buffer_capacity": 128,
                "standardize_start_time_step": 8,
                "end_exploration_time_step": 8,
                "action_scaling_coefficient": 0.5,
                "reward_scaling": 5.0,
                "update_per_time_step": 1,
                "normalized": False,
                "policy_state_dict": {},
                "soft_q1_state_dict": {},
                "soft_q2_state_dict": {},
                "target_soft_q1_state_dict": {},
                "target_soft_q2_state_dict": {},
                "policy_optimizer_state_dict": {},
                "soft_q_optimizer1_state_dict": {},
                "soft_q_optimizer2_state_dict": {},
                "norm_mean": None,
                "norm_std": None,
                "r_norm_mean": None,
                "r_norm_std": None,
                "shared_context_dimension": 4,
            },
        }

    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store.resolve_imported_checkpoint_path",
        fake_resolve_imported_checkpoint_path,
    )
    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store.safe_load_checkpoint_payload",
        fake_safe_load_checkpoint_payload,
    )

    client = TestClient(app)
    response = client.post(f"/api/artifacts/{artifact_id}/evaluate", json={})

    assert response.status_code == 400
    assert "control_mode" in response.json()["detail"]

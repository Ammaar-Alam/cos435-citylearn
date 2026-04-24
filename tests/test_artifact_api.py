from __future__ import annotations

import builtins
import importlib
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


def test_artifact_import_stores_extra_files_alongside_primary(tmp_path: Path) -> None:
    # Regression: Central PPO checkpoints need ``ppo_model.zip`` and
    # its companion sidecars co-located under a single ``artifact_id``. The
    # import endpoint's ``extra_files`` field is the only way to produce that
    # layout in one request; without this test, a future refactor could silently
    # drop companion uploads and re-break central-PPO evaluation.
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/artifacts/import",
        data={
            "artifact_kind": "checkpoint",
            "label": "central ppo bundle",
            "runner_id": "ppo_central_baseline",
        },
        # httpx TestClient repeats ``files`` tuple entries as form fields, so
        # both ``extra_files`` entries arrive at FastAPI as a list.
        files=[
            ("file", ("ppo_model.zip", b"fake-sb3-zip", "application/zip")),
            ("extra_files", ("vec_normalize.pkl", b"fake-vec", "application/octet-stream")),
            ("extra_files", ("topology.json", b"{}", "application/json")),
            (
                "extra_files",
                (
                    "checkpoint_metadata.json",
                    b"{\"variant\":\"central_baseline\"}",
                    "application/json",
                ),
            ),
        ],
    )

    assert response.status_code == 200, response.text
    artifact = response.json()
    artifact_id = artifact["artifact_id"]
    artifact_dir = settings.imported_artifacts_root / artifact_id
    assert (artifact_dir / "ppo_model.zip").exists()
    assert (artifact_dir / "ppo_model.zip").read_bytes() == b"fake-sb3-zip"
    # The companion file must land in the same directory, not its own UUID.
    assert (artifact_dir / "vec_normalize.pkl").exists()
    assert (artifact_dir / "vec_normalize.pkl").read_bytes() == b"fake-vec"
    assert (artifact_dir / "topology.json").exists()
    assert (artifact_dir / "checkpoint_metadata.json").exists()
    # ``file_path`` still points at the primary upload -- it's what the
    # artifact record and loaders key off. Companion files are discovered by
    # convention (sibling lookup).
    assert artifact["source_filename"] == "ppo_model.zip"


def test_artifact_import_rejects_bound_central_ppo_missing_required_sidecars(
    tmp_path: Path,
) -> None:
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/artifacts/import",
        data={
            "artifact_kind": "checkpoint",
            "label": "central ppo missing sidecars",
            "runner_id": "ppo_central_baseline",
        },
        files=[
            ("file", ("ppo_model.zip", b"fake-sb3-zip", "application/zip")),
            ("extra_files", ("vec_normalize.pkl", b"fake-vec", "application/octet-stream")),
        ],
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "topology.json" in detail
    assert "checkpoint_metadata.json" in detail


def test_artifact_import_single_file_still_works_without_extra_files(tmp_path: Path) -> None:
    # Backwards-compat: SAC checkpoints, shared-PPO checkpoints, and playback
    # JSONs all ship as a single file. Adding ``extra_files`` to the endpoint
    # must not break the single-file path.
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/artifacts/import",
        data={"artifact_kind": "checkpoint", "label": "solo sac"},
        files={"file": ("checkpoint.pt", b"fake-torch", "application/octet-stream")},
    )

    assert response.status_code == 200, response.text
    artifact_dir = settings.imported_artifacts_root / response.json()["artifact_id"]
    assert (artifact_dir / "checkpoint.pt").exists()
    assert not (artifact_dir / "vec_normalize.pkl").exists()


def test_artifact_import_rejects_duplicate_filenames_in_extra_files(tmp_path: Path) -> None:
    # If a user accidentally attaches the primary filename again as an extra,
    # the second write would silently clobber the first. Fail at 400 with a
    # message that names the offending file, so the UI can surface it.
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/artifacts/import",
        data={"artifact_kind": "checkpoint", "label": "dupe"},
        files=[
            ("file", ("ppo_model.zip", b"primary", "application/zip")),
            ("extra_files", ("ppo_model.zip", b"collision", "application/zip")),
        ],
    )

    assert response.status_code == 400
    assert "duplicate filename" in response.json()["detail"]
    assert "ppo_model.zip" in response.json()["detail"]


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

    def fake_load_sac_checkpoint_tools():
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

        def fake_validate_checkpoint_env_compatibility(*_args, **_kwargs):
            return None

        from cos435_citylearn.algorithms.sac.checkpoints import (
            validate_checkpoint_runner_compatibility,
        )

        return (
            fake_resolve_imported_checkpoint_path,
            fake_safe_load_checkpoint_payload,
            fake_validate_checkpoint_env_compatibility,
            validate_checkpoint_runner_compatibility,
        )

    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store._load_sac_checkpoint_tools",
        fake_load_sac_checkpoint_tools,
    )

    client = TestClient(app)
    response = client.post(f"/api/artifacts/{artifact_id}/evaluate", json={})

    assert response.status_code == 400
    assert "control_mode" in response.json()["detail"]


def test_artifact_evaluate_rejects_incompatible_ppo_checkpoint_before_queueing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Regression: a central PPO artifact sidecar that advertises
    # ``reward_v1`` must be rejected by the API preflight before the job
    # queues, matching the SAC preflight above. Previously ``build_evaluation_request``
    # had no PPO branch -- mismatched artifacts silently queued and blew up
    # hours later inside the worker.
    settings = build_test_settings(tmp_path)
    app = create_app(settings)
    artifact_id = "artifact_bad_ppo"
    artifact_dir = settings.imported_artifacts_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # The preflight resolves the file via artifact.json but we short-circuit
    # to the sidecar loader below, so the actual zip content is irrelevant.
    model_path = artifact_dir / "ppo_model.zip"
    model_path.write_bytes(b"fake-sb3-zip")
    (artifact_dir / "vec_normalize.pkl").write_bytes(b"fake-vec")
    write_json(
        artifact_dir / "artifact.json",
        {
            "artifact_id": artifact_id,
            "artifact_kind": "checkpoint",
            "label": "bad ppo checkpoint",
            "source_filename": "ppo_model.zip",
            "imported_at": "2026-04-20T00:00:00Z",
            "algorithm": "ppo",
            "runner_id": "ppo_central_baseline",
            "status": "evaluable",
            "file_path": str(model_path),
            "notes": None,
            "evaluable": True,
            "playback_path": None,
            "simulation_dir": None,
        },
    )

    def fake_load_central_ppo_sidecar_tools():
        def fake_load_sidecar(_model_path):
            # Sidecar reports the artifact was trained under reward_v1 /
            # features_v1 -- the runner config (ppo_central_baseline.yaml)
            # expects reward_v2 / features_v2. The real validator should raise.
            return {
                "algorithm": "ppo",
                "control_mode": "centralized",
                "variant": "central_baseline",
                "reward_version": "v1",
                "features_version": "v1",
            }

        from cos435_citylearn.baselines.ppo import _validate_central_ppo_sidecar

        return fake_load_sidecar, _validate_central_ppo_sidecar

    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store._load_central_ppo_sidecar_tools",
        fake_load_central_ppo_sidecar_tools,
    )

    client = TestClient(app)
    response = client.post(f"/api/artifacts/{artifact_id}/evaluate", json={})

    assert response.status_code == 400
    detail = response.json()["detail"]
    # The validator mentions the specific mismatched fields; make sure the
    # 400 payload surfaces something actionable rather than a generic error.
    assert "reward_version" in detail or "features_version" in detail


def test_artifact_evaluate_rejects_central_ppo_missing_vec_normalize_before_queueing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = build_test_settings(tmp_path)
    app = create_app(settings)
    artifact_id = "artifact_missing_vec"
    artifact_dir = settings.imported_artifacts_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "ppo_model.zip"
    model_path.write_bytes(b"fake-sb3-zip")
    write_json(
        artifact_dir / "artifact.json",
        {
            "artifact_id": artifact_id,
            "artifact_kind": "checkpoint",
            "label": "central ppo missing vec",
            "source_filename": "ppo_model.zip",
            "imported_at": "2026-04-22T00:00:00Z",
            "algorithm": "ppo",
            "runner_id": "ppo_central_baseline",
            "status": "evaluable",
            "file_path": str(model_path),
            "notes": None,
            "evaluable": True,
            "playback_path": None,
            "simulation_dir": None,
        },
    )

    def fake_load_central_ppo_sidecar_tools():
        def fake_load_sidecar(_model_path):
            return {
                "algorithm": "ppo",
                "control_mode": "centralized",
                "variant": "central_baseline",
                "reward_version": "reward_v0",
                "features_version": "base_central_obs",
            }

        def fake_validate_sidecar(*_args, **_kwargs):
            return {}

        return fake_load_sidecar, fake_validate_sidecar

    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store._load_central_ppo_sidecar_tools",
        fake_load_central_ppo_sidecar_tools,
    )

    client = TestClient(app)
    response = client.post(f"/api/artifacts/{artifact_id}/evaluate", json={})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "vec_normalize.pkl" in detail
    assert "centralized PPO cannot evaluate" in detail


def test_artifact_evaluate_rejects_central_ppo_missing_topology_before_queueing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = build_test_settings(tmp_path)
    app = create_app(settings)
    artifact_id = "artifact_missing_topology"
    artifact_dir = settings.imported_artifacts_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "ppo_model.zip"
    model_path.write_bytes(b"fake-sb3-zip")
    (artifact_dir / "vec_normalize.pkl").write_bytes(b"fake-vec")
    write_json(
        artifact_dir / "artifact.json",
        {
            "artifact_id": artifact_id,
            "artifact_kind": "checkpoint",
            "label": "central ppo missing topology",
            "source_filename": "ppo_model.zip",
            "imported_at": "2026-04-22T00:00:00Z",
            "algorithm": "ppo",
            "runner_id": "ppo_central_baseline",
            "status": "evaluable",
            "file_path": str(model_path),
            "notes": None,
            "evaluable": True,
            "playback_path": None,
            "simulation_dir": None,
        },
    )

    def fake_load_central_ppo_sidecar_tools():
        def fake_load_sidecar(_model_path):
            return {
                "algorithm": "ppo",
                "control_mode": "centralized",
                "variant": "central_baseline",
                "reward_version": "reward_v0",
                "features_version": "base_central_obs",
            }

        def fake_validate_sidecar(*_args, **_kwargs):
            return {}

        return fake_load_sidecar, fake_validate_sidecar

    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store._load_central_ppo_sidecar_tools",
        fake_load_central_ppo_sidecar_tools,
    )

    client = TestClient(app)
    response = client.post(f"/api/artifacts/{artifact_id}/evaluate", json={})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "topology.json" in detail
    assert "centralized PPO cannot evaluate safely" in detail


def test_artifact_evaluate_rejects_central_ppo_topology_mismatch_before_queueing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = build_test_settings(tmp_path)
    app = create_app(settings)
    artifact_id = "artifact_bad_topology"
    artifact_dir = settings.imported_artifacts_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "ppo_model.zip"
    model_path.write_bytes(b"fake-sb3-zip")
    (artifact_dir / "vec_normalize.pkl").write_bytes(b"fake-vec")
    write_json(
        artifact_dir / "topology.json",
        {
            "observation_names": [["hour", "load"]] * 3,
            "action_names": [["battery"]] * 3,
        },
    )
    write_json(
        artifact_dir / "artifact.json",
        {
            "artifact_id": artifact_id,
            "artifact_kind": "checkpoint",
            "label": "central ppo wrong split",
            "source_filename": "ppo_model.zip",
            "imported_at": "2026-04-22T00:00:00Z",
            "algorithm": "ppo",
            "runner_id": "ppo_central_baseline",
            "status": "evaluable",
            "file_path": str(model_path),
            "notes": None,
            "evaluable": True,
            "playback_path": None,
            "simulation_dir": None,
        },
    )

    def fake_load_ppo_checkpoint_tools():
        def fake_resolve_imported_checkpoint_path(**_kwargs):
            return model_path

        def fake_safe_load_checkpoint_payload(_path):
            raise AssertionError("central PPO preflight should not load torch payloads")

        def fake_validate_checkpoint_env_compatibility(*_args, **_kwargs):
            raise AssertionError("central PPO preflight should use topology metadata")

        def fake_validate_checkpoint_runner_compatibility(*_args, **_kwargs):
            raise AssertionError("central PPO preflight should use sidecar validation")

        return (
            fake_resolve_imported_checkpoint_path,
            fake_safe_load_checkpoint_payload,
            fake_validate_checkpoint_env_compatibility,
            fake_validate_checkpoint_runner_compatibility,
        )

    def fake_load_central_ppo_sidecar_tools():
        def fake_load_sidecar(_model_path):
            return {
                "algorithm": "ppo",
                "control_mode": "centralized",
                "variant": "central_baseline",
                "reward_version": "reward_v0",
                "features_version": "base_central_obs",
            }

        def fake_validate_sidecar(*_args, **_kwargs):
            return {}

        return fake_load_sidecar, fake_validate_sidecar

    def fake_make_citylearn_env(*_args, **_kwargs):
        return type(
            "Bundle",
            (),
            {
                "env": type(
                    "Env",
                    (),
                    {
                        "observation_names": [["hour", "load"]] * 6,
                        "action_names": [["battery"]] * 6,
                    },
                )(),
            },
        )()

    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store._load_ppo_checkpoint_tools",
        fake_load_ppo_checkpoint_tools,
    )
    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store._load_central_ppo_sidecar_tools",
        fake_load_central_ppo_sidecar_tools,
    )
    monkeypatch.setattr(
        "cos435_citylearn.api.services.artifact_store._load_citylearn_env_factory",
        lambda: fake_make_citylearn_env,
    )

    client = TestClient(app)
    response = client.post(f"/api/artifacts/{artifact_id}/evaluate", json={"split": "phase_3_1"})

    assert response.status_code == 400
    assert "trained on 3 buildings but target env has 6" in response.json()["detail"]


def test_create_app_does_not_require_sac_modules_for_startup(tmp_path: Path, monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "cos435_citylearn.algorithms.sac",
            "cos435_citylearn.env",
        }:
            raise ImportError(f"blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    import cos435_citylearn.api.app as app_module
    import cos435_citylearn.api.services.artifact_store as artifact_store_module

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(artifact_store_module)
    importlib.reload(app_module)

    try:
        app = app_module.create_app(build_test_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/api/system/health")
        assert response.status_code == 200
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(artifact_store_module)
        importlib.reload(app_module)

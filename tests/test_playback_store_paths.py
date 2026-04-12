from __future__ import annotations

import json
from pathlib import Path

from cos435_citylearn.api.services.playback_store import PlaybackStore
from cos435_citylearn.api.settings import ApiSettings


def test_artifact_playback_resolves_repo_relative_results_path(tmp_path: Path) -> None:
    repo_root = tmp_path
    results_root = tmp_path / "results"
    playback_path = results_root / "dashboard" / "artifacts" / "artifact_a" / "playback.json"
    playback_path.parent.mkdir(parents=True, exist_ok=True)
    playback_path.write_text(
        json.dumps(
            {
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
                "buildings": [
                    {
                        "name": "Building_1",
                        "series": {"battery_soc": [12.0, 18.0, 21.0]},
                    }
                ],
                "district": {"net_electricity_consumption": [1.0, 0.8, 0.6]},
                "timestamps": [
                    "2023-01-01T00:00:00Z",
                    "2023-01-01T01:00:00Z",
                    "2023-01-01T02:00:00Z",
                ],
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
        )
    )
    settings = ApiSettings(
        repo_root=repo_root,
        config_root=repo_root / "configs",
        results_root=results_root,
        run_root=results_root / "runs",
        manifests_root=results_root / "manifests",
        ui_exports_root=results_root / "ui_exports",
        jobs_root=results_root / "dashboard" / "jobs",
        imported_artifacts_root=results_root / "dashboard" / "artifacts",
        artifacts_root=results_root,
        frontend_root=repo_root / "apps" / "dashboard",
        frontend_dist=repo_root / "apps" / "dashboard" / "dist",
        python_executable=repo_root / ".venv" / "bin" / "python",
        mpl_config_dir=repo_root / ".cache" / "matplotlib",
        max_concurrent_jobs=1,
    )

    store = PlaybackStore(settings)
    playback = store.get_artifact_playback("results/dashboard/artifacts/artifact_a/playback.json")
    assert playback.run_id == "imported_run"
    assert playback.stored_steps == 3
    assert playback.total_steps == 3
    assert playback.truncated is False

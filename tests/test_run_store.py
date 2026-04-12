from __future__ import annotations

import json
from pathlib import Path

from cos435_citylearn.api.services.run_store import RunStore
from cos435_citylearn.api.settings import ApiSettings


def test_list_runs_does_not_load_rollout_trace_for_summary(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    results_root = tmp_path / "results"
    run_dir = results_root / "runs" / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    playback_manifest_path = run_dir / "playback_manifest.json"
    trace_path = run_dir / "rollout_trace.json"

    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "generated_at": "2026-04-12T00:00:00Z",
                "step_count": 719,
                "playback_path": "results/ui_exports/playback/run_a.json",
                "simulation_dir": None,
            }
        )
    )
    metrics_path.write_text(
        json.dumps(
            {
                "algorithm": "rbc",
                "variant": "basic_rbc",
                "split": "public_dev",
                "seed": 0,
                "dataset_name": "citylearn",
                "average_score": 1.023,
                "challenge_metrics": {},
                "district_kpis": {},
            }
        )
    )
    playback_manifest_path.write_text(
        json.dumps(
            {
                "decision_steps": 128,
                "media": {"gif_path": None, "poster_path": None},
            }
        )
    )
    trace_path.write_text('[{"step": 0}]')

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

    store = RunStore(settings)
    original_load_json = store._load_json

    def guarded_load_json(path: Path):
        if path == trace_path:
            raise AssertionError("rollout trace should not be loaded for run summaries")
        return original_load_json(path)

    monkeypatch.setattr(store, "_load_json", guarded_load_json)

    runs = store.list_runs()

    assert len(runs) == 1
    assert runs[0].run_id == "run_a"
    assert runs[0].artifacts["trace_steps"] == 128

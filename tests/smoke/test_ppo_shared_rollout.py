from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cos435_citylearn.baselines import run_ppo
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


@pytest.mark.smoke
def test_shared_ppo_smoke_rollout(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    payload = run_ppo(
        config_path="configs/train/ppo/ppo_shared_dtde_smoke.yaml",
        eval_config_path="configs/eval/sac_smoke.yaml",
        output_root=tmp_path / "runs",
        metrics_root=tmp_path / "metrics",
        manifests_root=tmp_path / "manifests",
        ui_exports_root=tmp_path / "ui_exports",
        artifacts_root=tmp_path,
    )

    run_dir = Path(payload["run_dir"])
    assert payload["average_score"] is not None
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "training_curve.csv").exists()
    assert (run_dir / "rollout_trace.json").exists()

    # make sure we actually did a PPO update
    with (run_dir / "training_curve.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "shared PPO smoke config must produce at least one training iteration"
    updates = [int(float(row["n_updates"] or 0)) for row in rows]
    assert max(updates) >= 1, f"shared PPO smoke must run at least one SGD update, got {updates}"


@pytest.mark.smoke
def test_shared_ppo_cross_topology_eval(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()

    train_payload = run_ppo(
        config_path="configs/train/ppo/ppo_shared_dtde_smoke.yaml",
        eval_config_path="configs/eval/sac_smoke.yaml",
        output_root=tmp_path / "runs",
        metrics_root=tmp_path / "metrics",
        manifests_root=tmp_path / "manifests",
        ui_exports_root=tmp_path / "ui_exports",
        artifacts_root=tmp_path,
    )
    trained_run_id = train_payload["run_id"]

    # swap to phase_3_1 for 3->6 cross-topology eval
    eval_payload = run_ppo(
        config_path="configs/train/ppo/ppo_shared_dtde_smoke.yaml",
        eval_config_path="configs/eval/sac_smoke.yaml",
        output_root=tmp_path / "runs",
        metrics_root=tmp_path / "metrics",
        manifests_root=tmp_path / "manifests",
        ui_exports_root=tmp_path / "ui_exports",
        artifacts_root=tmp_path,
        artifact_id=trained_run_id,
        split_override="phase_3_1",
    )

    run_dir = Path(eval_payload["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["split"] == "phase_3_1"
    assert manifest["control_mode"] == "shared_dtde"
    assert manifest["artifact_id"] == trained_run_id
    assert manifest.get("trained_on_split") == "public_dev"

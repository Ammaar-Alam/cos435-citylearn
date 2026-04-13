from __future__ import annotations

from pathlib import Path

import pytest

from cos435_citylearn.baselines import run_sac
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


@pytest.mark.smoke
def test_centralized_sac_smoke_rollout(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    payload = run_sac(
        config_path="configs/train/sac/sac_central_smoke.yaml",
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


@pytest.mark.smoke
def test_shared_sac_smoke_rollout(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    payload = run_sac(
        config_path="configs/train/sac/sac_shared_dtde_smoke.yaml",
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

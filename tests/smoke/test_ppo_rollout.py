from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cos435_citylearn.baselines import run_ppo
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


@pytest.mark.smoke
def test_centralized_ppo_smoke_rollout(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    payload = run_ppo(
        config_path="configs/train/ppo/ppo_central_smoke.yaml",
        eval_config_path="configs/eval/sac_smoke.yaml",
        output_root=tmp_path / "runs",
        metrics_root=tmp_path / "metrics",
        manifests_root=tmp_path / "manifests",
    )

    run_dir = Path(payload["run_dir"])
    assert payload["average_score"] is not None
    assert (run_dir / "ppo_model.zip").exists()
    assert (run_dir / "vec_normalize.pkl").exists()
    assert (run_dir / "training_curve.csv").exists()
    assert (run_dir / "rollout_trace.json").exists()


@pytest.mark.smoke
def test_ppo_rollout_trace_honors_trace_limit(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    config = yaml.safe_load(Path("configs/train/ppo/ppo_central_smoke.yaml").read_text())
    config["evaluation"]["trace_limit"] = 5
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    eval_config = yaml.safe_load(Path("configs/eval/sac_smoke.yaml").read_text())
    eval_config["evaluation"]["max_steps"] = 12
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))

    payload = run_ppo(
        config_path=config_path,
        eval_config_path=eval_path,
        output_root=tmp_path / "runs",
        metrics_root=tmp_path / "metrics",
        manifests_root=tmp_path / "manifests",
    )

    run_dir = Path(payload["run_dir"])
    trace = yaml.safe_load((run_dir / "rollout_trace.json").read_text())
    assert len(trace) == 5

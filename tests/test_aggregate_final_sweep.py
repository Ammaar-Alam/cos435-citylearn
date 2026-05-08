from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cos435_citylearn.paths import REPO_ROOT


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregate_final_sweep_fails_when_expected_cells_are_missing(tmp_path: Path) -> None:
    sweep_root = tmp_path / "final_sweep"
    summary_path = tmp_path / "summary.csv"
    sweep_root.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cluster" / "aggregate_final_sweep.py"),
            "--sweep-root",
            str(sweep_root),
            "--out",
            str(summary_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode != 0
    assert "MISSING CELLS" in result.stdout
    assert "sac_lr1e-4_hp-reward_scaling_val-2p5_seed0" in result.stdout


def test_aggregate_final_sweep_writes_present_cells_when_allowed(tmp_path: Path) -> None:
    sweep_root = tmp_path / "final_sweep"
    cell_id = "sac_lr1e-4_hp-reward_scaling_val-2p5_seed0"
    cell = sweep_root / cell_id
    summary_path = tmp_path / "summary.csv"

    _write_json(
        cell / "meta.json",
        {
            "cell_id": cell_id,
            "algo": "sac",
            "lr": "1e-4",
            "seed": 0,
            "hyperparameter": "reward_scaling",
            "hyperparameter_value": "2.5",
        },
    )
    _write_json(cell / "train.json", {"run_id": "train-run", "average_score": 0.5})
    _write_json(
        cell / "eval_phase_2_online_eval_1.json",
        {"run_id": "eval-run", "average_score": 0.6},
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cluster" / "aggregate_final_sweep.py"),
            "--sweep-root",
            str(sweep_root),
            "--out",
            str(summary_path),
            "--allow-missing",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    rows = summary_path.read_text().splitlines()
    assert (
        "sac_lr1e-4_hp-reward_scaling_val-2p5_seed0,sac,1e-4,0,"
        "reward_scaling,2.5,public_dev,train-run,0.5"
    ) in rows
    assert (
        "sac_lr1e-4_hp-reward_scaling_val-2p5_seed0,sac,1e-4,0,"
        "reward_scaling,2.5,phase_2_online_eval_1,eval-run,0.6"
    ) in rows

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


def test_aggregate_final_sweep_supports_mappo_only_matrix(tmp_path: Path) -> None:
    sweep_root = tmp_path / "mappo_sweep"
    cell_id = "mappo_lr3e-4_hp-ent_coef_val-0p01_seed2"
    cell = sweep_root / cell_id
    summary_path = tmp_path / "summary.csv"

    _write_json(
        cell / "meta.json",
        {
            "cell_id": cell_id,
            "algo": "mappo",
            "lr": "3e-4",
            "seed": 2,
            "hyperparameter": "ent_coef",
            "hyperparameter_value": "0.01",
        },
    )
    _write_json(cell / "train.json", {"run_id": "mappo-train", "average_score": 0.7})
    _write_json(
        cell / "eval_phase_3_1.json",
        {"run_id": "mappo-p3", "average_score": 0.8},
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cluster" / "aggregate_final_sweep.py"),
            "--algos",
            "mappo",
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
        "mappo_lr3e-4_hp-ent_coef_val-0p01_seed2,mappo,3e-4,2,"
        "ent_coef,0.01,public_dev,mappo-train,0.7"
    ) in rows
    assert (
        "mappo_lr3e-4_hp-ent_coef_val-0p01_seed2,mappo,3e-4,2,"
        "ent_coef,0.01,phase_3_1,mappo-p3,0.8"
    ) in rows

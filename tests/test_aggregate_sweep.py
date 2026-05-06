from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cos435_citylearn.paths import REPO_ROOT


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregate_sweep_ignores_unexpected_cells_for_missing_split_checks(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "sweep"
    expected = sweep_root / "ppo_lr1e-4_seed0"
    extra = sweep_root / "ppo_lr9e-4_seed99"

    _write_json(expected / "train.json", {"run_id": "expected-train", "average_score": 0.8})
    for split in ("phase_3_1", "phase_3_2", "phase_3_3"):
        _write_json(expected / f"eval_{split}.json", {"run_id": split, "average_score": 0.9})

    # Scratch dirs from ad-hoc reruns should not make the declared sweep fail.
    # This cell is outside the expected algo/lr/seed matrix and intentionally
    # lacks eval payloads.
    _write_json(extra / "train.json", {"run_id": "unexpected-train", "average_score": 1.0})

    summary_path = tmp_path / "summary.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cluster" / "aggregate_sweep.py"),
            "--sweep-root",
            str(sweep_root),
            "--out",
            str(summary_path),
            "--algos",
            "ppo",
            "--lrs",
            "1e-4",
            "--seeds",
            "0-0",
            "--eval-splits",
            "phase_3_1,phase_3_2,phase_3_3",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert summary_path.exists()
    assert "MISSING SPLITS" not in result.stdout


def test_aggregate_sweep_defaults_require_td3_cells_unless_allowed(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "sweep"
    (sweep_root / "ppo_lr1e-4_seed0").mkdir(parents=True)
    summary_path = tmp_path / "summary.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cluster" / "aggregate_sweep.py"),
            "--sweep-root",
            str(sweep_root),
            "--out",
            str(summary_path),
            "--lrs",
            "1e-4",
            "--seeds",
            "0-0",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode != 0
    assert "td3 lr=1e-4 seed=0" in result.stdout

    allowed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cluster" / "aggregate_sweep.py"),
            "--sweep-root",
            str(sweep_root),
            "--out",
            str(summary_path),
            "--lrs",
            "1e-4",
            "--seeds",
            "0-0",
            "--allow-missing",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert allowed.returncode == 0, allowed.stderr or allowed.stdout
    assert "td3 lr=1e-4 seed=0" in allowed.stdout


def test_aggregate_sweep_parses_td3_cells(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "sweep"
    cell = sweep_root / "td3_lr1e-4_seed0"

    _write_json(cell / "train.json", {"run_id": "td3-train", "average_score": 0.7})
    _write_json(cell / "eval_phase_3_1.json", {"run_id": "td3-eval", "average_score": 0.8})

    summary_path = tmp_path / "summary.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cluster" / "aggregate_sweep.py"),
            "--sweep-root",
            str(sweep_root),
            "--out",
            str(summary_path),
            "--algos",
            "td3",
            "--lrs",
            "1e-4",
            "--seeds",
            "0-0",
            "--eval-splits",
            "phase_3_1",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    rows = summary_path.read_text().splitlines()
    assert "td3,1e-4,0,public_dev,td3-train,0.7" in rows
    assert "td3,1e-4,0,phase_3_1,td3-eval,0.8" in rows

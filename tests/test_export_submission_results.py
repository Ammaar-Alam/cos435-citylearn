from __future__ import annotations

import csv
from pathlib import Path

from scripts.analysis import export_submission_results as export

METRIC_FIELDNAMES = [
    "run_id",
    "algorithm",
    "variant",
    "split",
    "seed",
    "dataset_name",
    "average_score",
    "district_cost_total",
    "district_carbon_emissions_total",
    "district_daily_peak_average",
    "district_discomfort_proportion",
    "district_one_minus_thermal_resilience_proportion",
]


def _write_metric(
    path: Path,
    *,
    run_id: str,
    algorithm: str,
    variant: str,
    split: str = "public_dev",
    seed: int = 0,
    average_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "algorithm": algorithm,
                "variant": variant,
                "split": split,
                "seed": seed,
                "dataset_name": "citylearn_2023_public_dev",
                "average_score": average_score,
                "district_cost_total": 1.0,
                "district_carbon_emissions_total": 1.0,
                "district_daily_peak_average": 1.0,
                "district_discomfort_proportion": 0.0,
                "district_one_minus_thermal_resilience_proportion": 0.0,
            }
        )


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell_id",
                "algo",
                "lr",
                "seed",
                "hyperparameter",
                "hyperparameter_value",
                "split",
                "run_id",
                "average_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_variant_label_prefers_algorithm_specific_label() -> None:
    assert export._variant_label("central_baseline", "td3") == "Centralized TD3 baseline"
    assert export._variant_label("central_baseline", "sac") == "Centralized SAC baseline"
    assert (
        export._variant_label("mappo_shared_ctde_reward_v2", "mappo")
        == "Shared CTDE MAPPO reward_v2"
    )


def test_load_shared_sweep_rows_includes_mappo_sweep_summary(
    tmp_path: Path, monkeypatch
) -> None:
    metrics_root = tmp_path / "results" / "metrics"
    monkeypatch.setattr(export, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(export, "METRICS_ROOT", metrics_root)

    ppo_run = "ppo__ppo_shared_dtde_reward_v2__public_dev__seed0__old"
    mappo_run = "mappo__mappo_shared_ctde_reward_v2__public_dev__seed0__best"
    _write_metric(
        metrics_root / f"{ppo_run}.csv",
        run_id=ppo_run,
        algorithm="ppo",
        variant="ppo_shared_dtde_reward_v2",
        average_score=0.7,
    )
    _write_metric(
        metrics_root / f"{mappo_run}.csv",
        run_id=mappo_run,
        algorithm="mappo",
        variant="mappo_shared_ctde_reward_v2",
        average_score=0.4,
    )
    _write_summary(
        tmp_path / "results" / "sweep" / "summary.csv",
        [
            {
                "cell_id": "ppo_lr1e-4_seed0",
                "algo": "ppo",
                "lr": "1e-4",
                "seed": 0,
                "hyperparameter": "",
                "hyperparameter_value": "",
                "split": "public_dev",
                "run_id": ppo_run,
                "average_score": 0.7,
            }
        ],
    )
    _write_summary(
        tmp_path / "results" / "mappo_sweep" / "summary.csv",
        [
            {
                "cell_id": "mappo_lr1e-4_hp-ent_coef_val-0p0_seed0",
                "algo": "mappo",
                "lr": "1e-4",
                "seed": 0,
                "hyperparameter": "ent_coef",
                "hyperparameter_value": "0.0",
                "split": "public_dev",
                "run_id": mappo_run,
                "average_score": 0.4,
            }
        ],
    )

    rows = export._load_shared_sweep_rows()

    assert [(row.metric.algorithm, row.metric.run_id) for row in rows] == [
        ("mappo", mappo_run),
        ("ppo", ppo_run),
    ]


def test_load_local_rows_uses_mappo_sweep_best_instead_of_latest_metric(
    tmp_path: Path, monkeypatch
) -> None:
    metrics_root = tmp_path / "results" / "metrics"
    monkeypatch.setattr(export, "METRICS_ROOT", metrics_root)

    rbc_run = "rbc__basic_rbc__public_dev__seed0__baseline"
    stale_mappo_run = "mappo__mappo_shared_ctde_reward_v2__public_dev__seed0__zzz"
    best_mappo_run = "mappo__mappo_shared_ctde_reward_v2__public_dev__seed0__aaa"
    _write_metric(
        metrics_root / f"{rbc_run}.csv",
        run_id=rbc_run,
        algorithm="rbc",
        variant="basic_rbc",
        average_score=1.0,
    )
    _write_metric(
        metrics_root / f"{stale_mappo_run}.csv",
        run_id=stale_mappo_run,
        algorithm="mappo",
        variant="mappo_shared_ctde_reward_v2",
        average_score=0.9,
    )
    _write_metric(
        metrics_root / f"{best_mappo_run}.csv",
        run_id=best_mappo_run,
        algorithm="mappo",
        variant="mappo_shared_ctde_reward_v2",
        average_score=0.4,
    )
    shared_rows = [
        export.SharedSweepRow(
            lr="1e-4",
            metric=export._read_metric_row(metrics_root / f"{best_mappo_run}.csv"),
        )
    ]

    _, local_rows = export._load_local_rows(shared_rows)

    selected = {
        (row.algorithm, row.variant, row.seed): row.run_id
        for row in local_rows
        if row.algorithm == "mappo"
    }
    assert selected[("mappo", "mappo_shared_ctde_reward_v2", 0)] == best_mappo_run

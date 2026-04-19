from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = REPO_ROOT / "results" / "metrics"
OUTPUT_ROOT = REPO_ROOT / "submission" / "results"

OFFICIAL_REFERENCES = [
    {
        "method_id": "official_rbc_public",
        "method_label": "Official RBC baseline (public)",
        "split_type": "official_public",
        "average_score": 1.085,
        "source_note": "CityLearn 2023 winning paper Table 1",
    },
    {
        "method_id": "official_rbc_private",
        "method_label": "Official RBC baseline (private)",
        "split_type": "official_private",
        "average_score": 1.124,
        "source_note": "CityLearn 2023 winning paper Table 1",
    },
    {
        "method_id": "official_chesca_public",
        "method_label": "Official CHESCA winner (public)",
        "split_type": "official_public",
        "average_score": 0.562,
        "source_note": "CityLearn 2023 winning paper Table 1",
    },
    {
        "method_id": "official_chesca_private",
        "method_label": "Official CHESCA winner (private)",
        "split_type": "official_private",
        "average_score": 0.565,
        "source_note": "CityLearn 2023 winning paper Table 1",
    },
    {
        "method_id": "official_chesca_star_private",
        "method_label": "Official CHESCA* post-deadline (private)",
        "split_type": "official_private",
        "average_score": 0.548,
        "source_note": "CityLearn 2023 winning paper Table 1",
    },
]


@dataclass(frozen=True)
class MetricRow:
    file_name: str
    run_id: str
    algorithm: str
    variant: str
    split: str
    seed: int
    dataset_name: str
    average_score: float
    district_cost_total: float
    district_carbon_emissions_total: float
    district_daily_peak_average: float
    district_discomfort_proportion: float
    district_one_minus_thermal_resilience_proportion: float


def _read_metric_row(path: Path) -> MetricRow:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))

    return MetricRow(
        file_name=path.name,
        run_id=row["run_id"],
        algorithm=row["algorithm"],
        variant=row["variant"],
        split=row["split"],
        seed=int(row["seed"]),
        dataset_name=row["dataset_name"],
        average_score=float(row["average_score"]),
        district_cost_total=float(row["district_cost_total"]),
        district_carbon_emissions_total=float(row["district_carbon_emissions_total"]),
        district_daily_peak_average=float(row["district_daily_peak_average"]),
        district_discomfort_proportion=float(row["district_discomfort_proportion"]),
        district_one_minus_thermal_resilience_proportion=float(
            row["district_one_minus_thermal_resilience_proportion"]
        ),
    )


def _latest_metric(pattern: str) -> MetricRow:
    matches = sorted(METRICS_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no metric rows matched {pattern}")
    return _read_metric_row(matches[-1])


def _load_local_rows() -> tuple[MetricRow, list[MetricRow]]:
    rbc_row = _latest_metric("rbc__basic_rbc__public_dev__seed0__*.csv")
    sac_rows = sorted(
        (_read_metric_row(path) for path in METRICS_ROOT.glob("sac__*.csv")),
        key=lambda row: row.average_score,
    )
    if not sac_rows:
        raise FileNotFoundError("no sac metrics found under results/metrics")
    return rbc_row, sac_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_local_results_rows(rbc_row: MetricRow, sac_rows: list[MetricRow]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "method_id": "local_rbc",
            "method_label": "Local RBC baseline",
            "status": "measured",
            "algorithm": rbc_row.algorithm,
            "variant": rbc_row.variant,
            "run_id": rbc_row.run_id,
            "score_source": rbc_row.file_name,
            "split": rbc_row.split,
            "seed_count": 1,
            "average_score": round(rbc_row.average_score, 6),
            "delta_vs_local_rbc": 0.0,
            "pct_improvement_vs_local_rbc": 0.0,
            "district_cost_total": round(rbc_row.district_cost_total, 6),
            "district_carbon_emissions_total": round(rbc_row.district_carbon_emissions_total, 6),
            "district_daily_peak_average": round(rbc_row.district_daily_peak_average, 6),
            "district_discomfort_proportion": round(rbc_row.district_discomfort_proportion, 6),
            "district_one_minus_thermal_resilience_proportion": round(
                rbc_row.district_one_minus_thermal_resilience_proportion, 6
            ),
            "notes": "local phase_2 evaluation baseline",
        },
        {
            "method_id": "ppo_baseline_missing",
            "method_label": "PPO baseline",
            "status": "missing_artifact",
            "algorithm": "ppo",
            "variant": "central_baseline",
            "run_id": "",
            "score_source": "",
            "split": "public_dev",
            "seed_count": 0,
            "average_score": "",
            "delta_vs_local_rbc": "",
            "pct_improvement_vs_local_rbc": "",
            "district_cost_total": "",
            "district_carbon_emissions_total": "",
            "district_daily_peak_average": "",
            "district_discomfort_proportion": "",
            "district_one_minus_thermal_resilience_proportion": "",
            "notes": "no PPO result artifact found locally yet",
        },
    ]

    for row in sac_rows:
        delta = rbc_row.average_score - row.average_score
        rows.append(
            {
                "method_id": row.run_id,
                "method_label": f"SAC {row.variant}",
                "status": "measured",
                "algorithm": row.algorithm,
                "variant": row.variant,
                "run_id": row.run_id,
                "score_source": row.file_name,
                "split": row.split,
                "seed_count": 1,
                "average_score": round(row.average_score, 6),
                "delta_vs_local_rbc": round(delta, 6),
                "pct_improvement_vs_local_rbc": round(delta / rbc_row.average_score * 100.0, 2),
                "district_cost_total": round(row.district_cost_total, 6),
                "district_carbon_emissions_total": round(row.district_carbon_emissions_total, 6),
                "district_daily_peak_average": round(row.district_daily_peak_average, 6),
                "district_discomfort_proportion": round(row.district_discomfort_proportion, 6),
                "district_one_minus_thermal_resilience_proportion": round(
                    row.district_one_minus_thermal_resilience_proportion, 6
                ),
                "notes": "single-seed local phase_2 evaluation run",
            }
        )
    return rows


def _build_sac_ablation_rows(rbc_row: MetricRow, sac_rows: list[MetricRow]) -> list[dict[str, object]]:
    central_baseline = next(row for row in sac_rows if row.variant == "central_baseline")
    best_row = min(sac_rows, key=lambda row: row.average_score)
    rows = []
    for row in sac_rows:
        rows.append(
            {
                "variant": row.variant,
                "run_id": row.run_id,
                "average_score": round(row.average_score, 6),
                "delta_vs_central_baseline": round(row.average_score - central_baseline.average_score, 6),
                "delta_vs_best_sac": round(row.average_score - best_row.average_score, 6),
                "delta_vs_local_rbc": round(row.average_score - rbc_row.average_score, 6),
                "district_cost_total": round(row.district_cost_total, 6),
                "district_carbon_emissions_total": round(row.district_carbon_emissions_total, 6),
                "district_daily_peak_average": round(row.district_daily_peak_average, 6),
                "district_discomfort_proportion": round(row.district_discomfort_proportion, 6),
                "district_one_minus_thermal_resilience_proportion": round(
                    row.district_one_minus_thermal_resilience_proportion, 6
                ),
            }
        )
    return rows


def _build_reference_rows() -> list[dict[str, object]]:
    return [
        {
            "method_id": item["method_id"],
            "method_label": item["method_label"],
            "split_type": item["split_type"],
            "average_score": item["average_score"],
            "source_note": item["source_note"],
        }
        for item in OFFICIAL_REFERENCES
    ]


def _write_status_markdown(rbc_row: MetricRow, sac_rows: list[MetricRow]) -> None:
    best_row = min(sac_rows, key=lambda row: row.average_score)
    central_baseline = next(row for row in sac_rows if row.variant == "central_baseline")
    shared_v2 = next(row for row in sac_rows if row.variant == "shared_dtde_reward_v2")
    reward_v1 = next(row for row in sac_rows if row.variant == "central_reward_v1")

    delta_vs_rbc = rbc_row.average_score - best_row.average_score
    pct_vs_rbc = delta_vs_rbc / rbc_row.average_score * 100.0

    text = f"""# Current Local Results Snapshot

These files are the clean tracked summary of the raw outputs under `results/`.

## What is currently measured

- local RBC baseline: `{rbc_row.average_score:.6f}`
- SAC central baseline: `{central_baseline.average_score:.6f}`
- SAC central reward_v1: `{reward_v1.average_score:.6f}`
- SAC central reward_v2: `{best_row.average_score:.6f}`
- SAC shared dtde reward_v2: `{shared_v2.average_score:.6f}`
- PPO baseline artifact: missing locally

Lower is better.

## Current headline

The best local SAC run is `{best_row.variant}` at `{best_row.average_score:.6f}`.
That is `{delta_vs_rbc:.6f}` lower than the local RBC baseline, a
`{pct_vs_rbc:.2f}%` improvement on the local phase-2 evaluation dataset.

## What the current SAC ladder suggests

- reward shaping helped centralized SAC
- `reward_v2` beat `reward_v1`
- the current shared/decentralized `reward_v2` run is worse than the best centralized run
- these are still single-seed measurements, so they are not claim-quality yet

## Important caveats

- these numbers are local phase-2 evaluation numbers, not official leaderboard results
- there is still no saved PPO artifact in this checkout, so PPO vs SAC is not yet empirical
- all SAC rows here are from `seed 0` only
- final claims still need multi-seed and held-out evaluation
"""
    (OUTPUT_ROOT / "README.md").write_text(text)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rbc_row, sac_rows = _load_local_rows()

    local_rows = _build_local_results_rows(rbc_row, sac_rows)
    ablation_rows = _build_sac_ablation_rows(rbc_row, sac_rows)
    reference_rows = _build_reference_rows()

    _write_csv(
        OUTPUT_ROOT / "local_main_results.csv",
        [
            "method_id",
            "method_label",
            "status",
            "algorithm",
            "variant",
            "run_id",
            "score_source",
            "split",
            "seed_count",
            "average_score",
            "delta_vs_local_rbc",
            "pct_improvement_vs_local_rbc",
            "district_cost_total",
            "district_carbon_emissions_total",
            "district_daily_peak_average",
            "district_discomfort_proportion",
            "district_one_minus_thermal_resilience_proportion",
            "notes",
        ],
        local_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "sac_ablation_summary.csv",
        [
            "variant",
            "run_id",
            "average_score",
            "delta_vs_central_baseline",
            "delta_vs_best_sac",
            "delta_vs_local_rbc",
            "district_cost_total",
            "district_carbon_emissions_total",
            "district_daily_peak_average",
            "district_discomfort_proportion",
            "district_one_minus_thermal_resilience_proportion",
        ],
        ablation_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "official_benchmark_reference.csv",
        ["method_id", "method_label", "split_type", "average_score", "source_note"],
        reference_rows,
    )
    _write_status_markdown(rbc_row, sac_rows)


if __name__ == "__main__":
    main()

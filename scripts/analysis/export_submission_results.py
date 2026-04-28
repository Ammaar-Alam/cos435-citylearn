from __future__ import annotations

import csv
import math
import statistics as st
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = REPO_ROOT / "results" / "metrics"
OUTPUT_ROOT = REPO_ROOT / "submission" / "results"

TRACKED_OUTPUT_FILES = [
    "local_main_results.csv",
    "sac_ablation_summary.csv",
    "sac_seed_inventory.csv",
    "released_eval_main_results.csv",
    "released_eval_seed_inventory.csv",
    "ppo_shared_sweep_summary.csv",
    "ppo_shared_sweep_inventory.csv",
    "official_benchmark_reference.csv",
    "cross_split_scores.csv",
    "figure_manifest.csv",
    "method_comparison.csv",
    "README.md",
]

T_CRITICAL_95 = {
    2: 12.706205,
    3: 4.302653,
    4: 3.182446,
    5: 2.776445,
    6: 2.570582,
    7: 2.446912,
    8: 2.364624,
    9: 2.306004,
    10: 2.262157,
}

VARIANT_LABELS = {
    "central_baseline": "Centralized SAC baseline",
    "central_reward_v1": "Centralized SAC reward_v1",
    "central_reward_v2": "Centralized SAC reward_v2",
    "shared_dtde_reward_v2": "Shared DTDE SAC reward_v2",
    "ppo_central_baseline": "Centralized PPO baseline",
    "ppo_shared_dtde_reward_v2": "Shared DTDE PPO reward_v2",
}
REQUIRED_LOCAL_SAC_VARIANTS = (
    "central_baseline",
    "central_reward_v1",
    "central_reward_v2",
    "shared_dtde_reward_v2",
)

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


@dataclass(frozen=True)
class MetricSummary:
    algorithm: str
    variant: str
    split: str
    dataset_name: str
    rows: tuple[MetricRow, ...]
    average_score_mean: float
    average_score_std: float
    average_score_ci95: float
    best_average_score: float
    worst_average_score: float
    district_cost_total_mean: float
    district_carbon_emissions_total_mean: float
    district_daily_peak_average_mean: float
    district_discomfort_proportion_mean: float
    district_one_minus_thermal_resilience_proportion_mean: float


@dataclass(frozen=True)
class ReleasedSummary:
    algorithm: str
    variant: str
    scope: str
    split_names: tuple[str, ...]
    dataset_names: tuple[str, ...]
    rows: tuple[MetricRow, ...]
    average_score_mean: float
    average_score_std: float
    average_score_ci95: float
    best_average_score: float
    worst_average_score: float
    district_cost_total_mean: float
    district_carbon_emissions_total_mean: float
    district_daily_peak_average_mean: float
    district_discomfort_proportion_mean: float
    district_one_minus_thermal_resilience_proportion_mean: float


@dataclass(frozen=True)
class PpoSweepRow:
    lr: str
    metric: MetricRow


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
    latest_rows: dict[tuple[str, str, int], MetricRow] = {}
    for path in sorted(METRICS_ROOT.glob("sac__*.csv")):
        if "__public_dev__" not in path.name:
            continue
        row = _read_metric_row(path)
        latest_rows[(row.variant, row.split, row.seed)] = row
    sac_rows = sorted(
        latest_rows.values(),
        key=lambda row: (row.variant, row.seed, row.run_id),
    )
    if not sac_rows:
        raise FileNotFoundError("no sac metrics found under results/metrics")
    return rbc_row, sac_rows


def _released_group(split: str) -> str:
    if split.startswith("phase_2_online_eval_"):
        return "released_phase_2_online_eval"
    if split.startswith("phase_3_"):
        return "released_phase_3"
    raise ValueError(f"unsupported released split: {split}")


def _load_released_rows() -> list[MetricRow]:
    latest_rows: dict[tuple[str, str, int], MetricRow] = {}
    for path in sorted(METRICS_ROOT.glob("sac__*.csv")):
        row = _read_metric_row(path)
        if not row.split.startswith("phase_"):
            continue
        latest_rows[(row.variant, row.split, row.seed)] = row

    return sorted(
        latest_rows.values(),
        key=lambda row: (_released_group(row.split), row.variant, row.split, row.seed, row.run_id),
    )


def _load_ppo_sweep_rows() -> list[PpoSweepRow]:
    sweep_summary_path = REPO_ROOT / "results" / "sweep" / "summary.csv"
    if not sweep_summary_path.exists():
        return []

    rows: list[PpoSweepRow] = []
    with sweep_summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["algo"] != "ppo":
                continue
            run_id = row["run_id"]
            metric_path = METRICS_ROOT / f"{run_id}.csv"
            if not metric_path.exists():
                raise FileNotFoundError(
                    f"missing metric row for PPO sweep run_id {run_id}: expected {metric_path}"
                )
            rows.append(PpoSweepRow(lr=row["lr"], metric=_read_metric_row(metric_path)))

    return sorted(
        rows,
        key=lambda item: (
            item.lr,
            item.metric.split,
            item.metric.seed,
            item.metric.run_id,
        ),
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return st.fmean(values)


def _std(values: list[float]) -> float:
    return st.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    critical = T_CRITICAL_95.get(len(values), 1.96)
    return critical * _std(values) / math.sqrt(len(values))


def _evidence_level(seed_count: int) -> str:
    if seed_count >= 5:
        return "claim_run"
    if seed_count >= 3:
        return "pilot"
    if seed_count == 2:
        return "pilot"
    return "single_seed"


def _variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " "))


def _summarize_rows(rows: list[MetricRow]) -> MetricSummary:
    ordered_rows = tuple(sorted(rows, key=lambda row: row.seed))
    scores = [row.average_score for row in ordered_rows]
    costs = [row.district_cost_total for row in ordered_rows]
    carbons = [row.district_carbon_emissions_total for row in ordered_rows]
    peaks = [row.district_daily_peak_average for row in ordered_rows]
    discomforts = [row.district_discomfort_proportion for row in ordered_rows]
    resiliences = [
        row.district_one_minus_thermal_resilience_proportion for row in ordered_rows
    ]

    return MetricSummary(
        algorithm=ordered_rows[0].algorithm,
        variant=ordered_rows[0].variant,
        split=ordered_rows[0].split,
        dataset_name=ordered_rows[0].dataset_name,
        rows=ordered_rows,
        average_score_mean=_mean(scores),
        average_score_std=_std(scores),
        average_score_ci95=_ci95(scores),
        best_average_score=min(scores),
        worst_average_score=max(scores),
        district_cost_total_mean=_mean(costs),
        district_carbon_emissions_total_mean=_mean(carbons),
        district_daily_peak_average_mean=_mean(peaks),
        district_discomfort_proportion_mean=_mean(discomforts),
        district_one_minus_thermal_resilience_proportion_mean=_mean(resiliences),
    )


def _summarize_released_rows(scope: str, rows: list[MetricRow]) -> ReleasedSummary:
    ordered_rows = tuple(sorted(rows, key=lambda row: (row.split, row.seed, row.run_id)))
    scores = [row.average_score for row in ordered_rows]
    costs = [row.district_cost_total for row in ordered_rows]
    carbons = [row.district_carbon_emissions_total for row in ordered_rows]
    peaks = [row.district_daily_peak_average for row in ordered_rows]
    discomforts = [row.district_discomfort_proportion for row in ordered_rows]
    resiliences = [
        row.district_one_minus_thermal_resilience_proportion for row in ordered_rows
    ]

    return ReleasedSummary(
        algorithm=ordered_rows[0].algorithm,
        variant=ordered_rows[0].variant,
        scope=scope,
        split_names=tuple(sorted({row.split for row in ordered_rows})),
        dataset_names=tuple(sorted({row.dataset_name for row in ordered_rows})),
        rows=ordered_rows,
        average_score_mean=_mean(scores),
        average_score_std=_std(scores),
        average_score_ci95=_ci95(scores),
        best_average_score=min(scores),
        worst_average_score=max(scores),
        district_cost_total_mean=_mean(costs),
        district_carbon_emissions_total_mean=_mean(carbons),
        district_daily_peak_average_mean=_mean(peaks),
        district_discomfort_proportion_mean=_mean(discomforts),
        district_one_minus_thermal_resilience_proportion_mean=_mean(resiliences),
    )


def _build_sac_summaries(sac_rows: list[MetricRow]) -> list[MetricSummary]:
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in sac_rows:
        grouped.setdefault((row.variant, row.split), []).append(row)

    return sorted(
        (_summarize_rows(rows) for rows in grouped.values()),
        key=lambda summary: (summary.average_score_mean, summary.variant),
    )


def _build_released_group_summaries(rows: list[MetricRow]) -> list[ReleasedSummary]:
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.variant, _released_group(row.split)), []).append(row)

    summaries = (
        _summarize_released_rows(scope, grouped_rows)
        for (_, scope), grouped_rows in grouped.items()
    )
    return sorted(
        summaries,
        key=lambda summary: (summary.scope, summary.average_score_mean, summary.variant),
    )


def _build_released_split_summaries(rows: list[MetricRow]) -> list[ReleasedSummary]:
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.variant, row.split), []).append(row)

    summaries = (
        _summarize_released_rows(scope, grouped_rows)
        for (_, scope), grouped_rows in grouped.items()
    )
    return sorted(
        summaries,
        key=lambda summary: (summary.scope, summary.average_score_mean, summary.variant),
    )


def _build_local_results_rows(
    rbc_row: MetricRow, sac_summaries: list[MetricSummary]
) -> list[dict[str, object]]:
    rbc_score = rbc_row.average_score
    rows: list[dict[str, object]] = [
        {
            "method_id": "local_rbc",
            "method_label": "Local RBC baseline",
            "status": "measured",
            "evidence_level": "baseline",
            "algorithm": rbc_row.algorithm,
            "variant": rbc_row.variant,
            "split": rbc_row.split,
            "seed_count": 1,
            "average_score_mean": round(rbc_score, 6),
            "average_score_std": 0.0,
            "average_score_ci95": 0.0,
            "best_average_score": round(rbc_score, 6),
            "worst_average_score": round(rbc_score, 6),
            "delta_vs_local_rbc_mean": 0.0,
            "pct_improvement_vs_local_rbc_mean": 0.0,
            "district_cost_total_mean": round(rbc_row.district_cost_total, 6),
            "district_carbon_emissions_total_mean": round(
                rbc_row.district_carbon_emissions_total, 6
            ),
            "district_daily_peak_average_mean": round(
                rbc_row.district_daily_peak_average, 6
            ),
            "district_discomfort_proportion_mean": round(
                rbc_row.district_discomfort_proportion, 6
            ),
            "district_one_minus_thermal_resilience_proportion_mean": round(
                rbc_row.district_one_minus_thermal_resilience_proportion, 6
            ),
            "best_run_id": rbc_row.run_id,
            "worst_run_id": rbc_row.run_id,
            "notes": "single-seed local phase_2 evaluation baseline",
        },
        {
            "method_id": "ppo_baseline_missing",
            "method_label": "PPO baseline",
            "status": "missing_artifact",
            "evidence_level": "missing",
            "algorithm": "ppo",
            "variant": "central_baseline",
            "split": "public_dev",
            "seed_count": 0,
            "average_score_mean": "",
            "average_score_std": "",
            "average_score_ci95": "",
            "best_average_score": "",
            "worst_average_score": "",
            "delta_vs_local_rbc_mean": "",
            "pct_improvement_vs_local_rbc_mean": "",
            "district_cost_total_mean": "",
            "district_carbon_emissions_total_mean": "",
            "district_daily_peak_average_mean": "",
            "district_discomfort_proportion_mean": "",
            "district_one_minus_thermal_resilience_proportion_mean": "",
            "best_run_id": "",
            "worst_run_id": "",
            "notes": "no PPO result artifact found locally yet",
        },
    ]

    for summary in sac_summaries:
        delta = rbc_score - summary.average_score_mean
        evidence_level = _evidence_level(len(summary.rows))
        if evidence_level == "claim_run":
            note = "multi-seed local phase_2 claim run"
        elif evidence_level == "pilot":
            note = "multi-seed local phase_2 pilot run"
        else:
            note = "single-seed local phase_2 evaluation run"

        best_row = min(summary.rows, key=lambda row: row.average_score)
        worst_row = max(summary.rows, key=lambda row: row.average_score)

        rows.append(
            {
                "method_id": f"sac_{summary.variant}_{summary.split}",
                "method_label": _variant_label(summary.variant),
                "status": "measured",
                "evidence_level": evidence_level,
                "algorithm": summary.algorithm,
                "variant": summary.variant,
                "split": summary.split,
                "seed_count": len(summary.rows),
                "average_score_mean": round(summary.average_score_mean, 6),
                "average_score_std": round(summary.average_score_std, 6),
                "average_score_ci95": round(summary.average_score_ci95, 6),
                "best_average_score": round(summary.best_average_score, 6),
                "worst_average_score": round(summary.worst_average_score, 6),
                "delta_vs_local_rbc_mean": round(delta, 6),
                "pct_improvement_vs_local_rbc_mean": round(
                    delta / rbc_score * 100.0, 2
                ),
                "district_cost_total_mean": round(summary.district_cost_total_mean, 6),
                "district_carbon_emissions_total_mean": round(
                    summary.district_carbon_emissions_total_mean, 6
                ),
                "district_daily_peak_average_mean": round(
                    summary.district_daily_peak_average_mean, 6
                ),
                "district_discomfort_proportion_mean": round(
                    summary.district_discomfort_proportion_mean, 6
                ),
                "district_one_minus_thermal_resilience_proportion_mean": round(
                    summary.district_one_minus_thermal_resilience_proportion_mean, 6
                ),
                "best_run_id": best_row.run_id,
                "worst_run_id": worst_row.run_id,
                "notes": note,
            }
        )
    return rows


def _build_sac_ablation_rows(
    rbc_row: MetricRow, sac_summaries: list[MetricSummary]
) -> list[dict[str, object]]:
    by_variant = {summary.variant: summary for summary in sac_summaries}
    central_baseline = by_variant["central_baseline"]

    rows = []
    for summary in sac_summaries:
        delta_vs_baseline = (
            central_baseline.average_score_mean - summary.average_score_mean
        )
        rows.append(
            {
                "variant": summary.variant,
                "method_label": _variant_label(summary.variant),
                "seed_count": len(summary.rows),
                "evidence_level": _evidence_level(len(summary.rows)),
                "average_score_mean": round(summary.average_score_mean, 6),
                "average_score_std": round(summary.average_score_std, 6),
                "average_score_ci95": round(summary.average_score_ci95, 6),
                "best_average_score": round(summary.best_average_score, 6),
                "worst_average_score": round(summary.worst_average_score, 6),
                "delta_vs_central_baseline_mean": round(delta_vs_baseline, 6),
                "pct_improvement_vs_central_baseline_mean": round(
                    delta_vs_baseline / central_baseline.average_score_mean * 100.0, 2
                ),
                "delta_vs_local_rbc_mean": round(
                    rbc_row.average_score - summary.average_score_mean, 6
                ),
                "district_cost_total_mean": round(summary.district_cost_total_mean, 6),
                "district_carbon_emissions_total_mean": round(
                    summary.district_carbon_emissions_total_mean, 6
                ),
                "district_daily_peak_average_mean": round(
                    summary.district_daily_peak_average_mean, 6
                ),
                "district_discomfort_proportion_mean": round(
                    summary.district_discomfort_proportion_mean, 6
                ),
                "district_one_minus_thermal_resilience_proportion_mean": round(
                    summary.district_one_minus_thermal_resilience_proportion_mean, 6
                ),
            }
        )
    return rows


def _build_seed_inventory_rows(sac_rows: list[MetricRow]) -> list[dict[str, object]]:
    rows = []
    for row in sorted(sac_rows, key=lambda item: (item.variant, item.seed, item.run_id)):
        rows.append(
            {
                "run_id": row.run_id,
                "file_name": row.file_name,
                "algorithm": row.algorithm,
                "variant": row.variant,
                "split": row.split,
                "seed": row.seed,
                "dataset_name": row.dataset_name,
                "average_score": round(row.average_score, 6),
                "district_cost_total": round(row.district_cost_total, 6),
                "district_carbon_emissions_total": round(
                    row.district_carbon_emissions_total, 6
                ),
                "district_daily_peak_average": round(
                    row.district_daily_peak_average, 6
                ),
                "district_discomfort_proportion": round(
                    row.district_discomfort_proportion, 6
                ),
                "district_one_minus_thermal_resilience_proportion": round(
                    row.district_one_minus_thermal_resilience_proportion, 6
                ),
            }
        )
    return rows


def _build_released_main_rows(
    released_group_summaries: list[ReleasedSummary],
) -> list[dict[str, object]]:
    phase_2_summaries = [
        summary
        for summary in released_group_summaries
        if summary.scope == "released_phase_2_online_eval"
    ]
    phase_2_winner = (
        min(phase_2_summaries, key=lambda summary: summary.average_score_mean)
        if phase_2_summaries
        else None
    )

    rows: list[dict[str, object]] = []
    for summary in released_group_summaries:
        delta_vs_phase_2_winner = ""
        if phase_2_winner is not None and summary.scope == "released_phase_2_online_eval":
            delta_vs_phase_2_winner = round(
                summary.average_score_mean - phase_2_winner.average_score_mean, 6
            )

        if summary.scope == "released_phase_2_online_eval":
            if phase_2_winner is not None and summary.variant == phase_2_winner.variant:
                note = "best released phase_2 result among saved checkpoints"
            else:
                note = "released phase_2 checkpoint evaluation summary"
        else:
            note = "released phase_3 checkpoint evaluation summary"

        rows.append(
            {
                "method_id": f"sac_{summary.variant}_{summary.scope}",
                "method_label": _variant_label(summary.variant),
                "algorithm": summary.algorithm,
                "variant": summary.variant,
                "eval_group": summary.scope,
                "split_count": len(summary.split_names),
                "eval_job_count": len(summary.rows),
                "seed_count": len({row.seed for row in summary.rows}),
                "average_score_mean": round(summary.average_score_mean, 6),
                "average_score_std": round(summary.average_score_std, 6),
                "average_score_ci95": round(summary.average_score_ci95, 6),
                "best_average_score": round(summary.best_average_score, 6),
                "worst_average_score": round(summary.worst_average_score, 6),
                "delta_vs_released_phase2_winner_mean": delta_vs_phase_2_winner,
                "district_cost_total_mean": round(summary.district_cost_total_mean, 6),
                "district_carbon_emissions_total_mean": round(
                    summary.district_carbon_emissions_total_mean, 6
                ),
                "district_daily_peak_average_mean": round(
                    summary.district_daily_peak_average_mean, 6
                ),
                "district_discomfort_proportion_mean": round(
                    summary.district_discomfort_proportion_mean, 6
                ),
                "district_one_minus_thermal_resilience_proportion_mean": round(
                    summary.district_one_minus_thermal_resilience_proportion_mean, 6
                ),
                "notes": note,
            }
        )

    return rows


def _build_released_seed_inventory_rows(
    released_rows: list[MetricRow],
) -> list[dict[str, object]]:
    rows = []
    for row in sorted(
        released_rows, key=lambda item: (item.variant, item.split, item.seed, item.run_id)
    ):
        rows.append(
            {
                "run_id": row.run_id,
                "file_name": row.file_name,
                "algorithm": row.algorithm,
                "variant": row.variant,
                "split": row.split,
                "eval_group": _released_group(row.split),
                "seed": row.seed,
                "dataset_name": row.dataset_name,
                "average_score": round(row.average_score, 6),
                "district_cost_total": round(row.district_cost_total, 6),
                "district_carbon_emissions_total": round(
                    row.district_carbon_emissions_total, 6
                ),
                "district_daily_peak_average": round(
                    row.district_daily_peak_average, 6
                ),
                "district_discomfort_proportion": round(
                    row.district_discomfort_proportion, 6
                ),
                "district_one_minus_thermal_resilience_proportion": round(
                    row.district_one_minus_thermal_resilience_proportion, 6
                ),
            }
        )
    return rows


def _build_ppo_sweep_summary_rows(ppo_rows: list[PpoSweepRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[PpoSweepRow]] = {}
    for row in ppo_rows:
        grouped.setdefault((row.lr, row.metric.split), []).append(row)

    rows: list[dict[str, object]] = []
    for (lr, split), grouped_rows in sorted(grouped.items()):
        summary = _summarize_rows([row.metric for row in grouped_rows])
        best_row = min(grouped_rows, key=lambda row: row.metric.average_score).metric
        worst_row = max(grouped_rows, key=lambda row: row.metric.average_score).metric
        if split == "public_dev":
            note = "10-seed shared PPO sweep on local phase_2"
        else:
            note = "10-seed shared PPO checkpoint evaluation on released phase_3"

        rows.append(
            {
                "method_id": f"ppo_{summary.variant}_{lr}_{split}",
                "method_label": _variant_label(summary.variant),
                "algorithm": summary.algorithm,
                "variant": summary.variant,
                "lr": lr,
                "split": split,
                "seed_count": len(summary.rows),
                "evidence_level": _evidence_level(len(summary.rows)),
                "average_score_mean": round(summary.average_score_mean, 6),
                "average_score_std": round(summary.average_score_std, 6),
                "average_score_ci95": round(summary.average_score_ci95, 6),
                "best_average_score": round(summary.best_average_score, 6),
                "worst_average_score": round(summary.worst_average_score, 6),
                "district_cost_total_mean": round(summary.district_cost_total_mean, 6),
                "district_carbon_emissions_total_mean": round(
                    summary.district_carbon_emissions_total_mean, 6
                ),
                "district_daily_peak_average_mean": round(
                    summary.district_daily_peak_average_mean, 6
                ),
                "district_discomfort_proportion_mean": round(
                    summary.district_discomfort_proportion_mean, 6
                ),
                "district_one_minus_thermal_resilience_proportion_mean": round(
                    summary.district_one_minus_thermal_resilience_proportion_mean, 6
                ),
                "best_run_id": best_row.run_id,
                "worst_run_id": worst_row.run_id,
                "notes": note,
            }
        )

    return rows


def _build_ppo_sweep_inventory_rows(ppo_rows: list[PpoSweepRow]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in ppo_rows:
        metric = row.metric
        rows.append(
            {
                "run_id": metric.run_id,
                "file_name": metric.file_name,
                "algorithm": metric.algorithm,
                "variant": metric.variant,
                "lr": row.lr,
                "split": metric.split,
                "seed": metric.seed,
                "dataset_name": metric.dataset_name,
                "average_score": round(metric.average_score, 6),
                "district_cost_total": round(metric.district_cost_total, 6),
                "district_carbon_emissions_total": round(
                    metric.district_carbon_emissions_total, 6
                ),
                "district_daily_peak_average": round(
                    metric.district_daily_peak_average, 6
                ),
                "district_discomfort_proportion": round(
                    metric.district_discomfort_proportion, 6
                ),
                "district_one_minus_thermal_resilience_proportion": round(
                    metric.district_one_minus_thermal_resilience_proportion, 6
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


def _write_status_markdown(
    rbc_row: MetricRow,
    sac_summaries: list[MetricSummary],
    released_group_summaries: list[ReleasedSummary],
    released_split_summaries: list[ReleasedSummary],
    ppo_sweep_rows: list[PpoSweepRow],
    ppo_sweep_summary_rows: list[dict[str, object]],
) -> None:
    by_variant = {summary.variant: summary for summary in sac_summaries}
    best_summary = min(sac_summaries, key=lambda summary: summary.average_score_mean)
    best_claim_summary = min(
        (summary for summary in sac_summaries if len(summary.rows) >= 5),
        key=lambda summary: summary.average_score_mean,
    )
    rbc_score = rbc_row.average_score
    best_claim_delta = rbc_score - best_claim_summary.average_score_mean
    best_claim_pct = best_claim_delta / rbc_score * 100.0
    reward_v1 = by_variant["central_reward_v1"]
    reward_v2 = by_variant["central_reward_v2"]
    released_by_key = {
        (summary.variant, summary.scope): summary for summary in released_group_summaries
    }
    released_split_by_key = {
        (summary.variant, summary.scope): summary for summary in released_split_summaries
    }
    released_phase_2 = [
        summary
        for summary in released_group_summaries
        if summary.scope == "released_phase_2_online_eval"
    ]
    released_phase_2_winner = (
        min(released_phase_2, key=lambda summary: summary.average_score_mean)
        if released_phase_2
        else None
    )
    shared_phase_3 = released_by_key.get(("shared_dtde_reward_v2", "released_phase_3"))

    if len(best_summary.rows) >= 5:
        headline_intro = (
            f"The best current mean is `{best_summary.variant}` at "
            f"`{best_summary.average_score_mean:.6f}` across `{len(best_summary.rows)}` seeds."
        )
    else:
        headline_intro = (
            f"The best pilot mean right now is `{best_summary.variant}` at "
            f"`{best_summary.average_score_mean:.6f}`, but that result only has "
            f"`{len(best_summary.rows)}` seeds."
        )

    def line(summary: MetricSummary) -> str:
        return (
            f"- {_variant_label(summary.variant)}: mean `{summary.average_score_mean:.6f}`, std "
            f"`{summary.average_score_std:.6f}`, 95% CI `{summary.average_score_ci95:.6f}`, "
            f"seeds `{len(summary.rows)}`"
        )

    def released_line(summary: ReleasedSummary) -> str:
        return (
            f"- {_variant_label(summary.variant)} on `{summary.scope}`: mean "
            f"`{summary.average_score_mean:.6f}`, std `{summary.average_score_std:.6f}`, "
            f"95% CI `{summary.average_score_ci95:.6f}`, eval jobs `{len(summary.rows)}`, "
            f"seeds `{len({row.seed for row in summary.rows})}`"
        )

    released_phase_2_text = ""
    if released_phase_2:
        released_phase_2_text = "\n".join(released_line(summary) for summary in released_phase_2)

    released_split_lines: list[str] = []
    for split in ("phase_2_online_eval_1", "phase_2_online_eval_2", "phase_2_online_eval_3"):
        central = released_split_by_key.get(("central_reward_v2", split))
        shared = released_split_by_key.get(("shared_dtde_reward_v2", split))
        if central is None or shared is None:
            continue
        released_split_lines.append(
            f"- `{split}`: central `reward_v2` `{central.average_score_mean:.6f}` vs "
            f"shared `reward_v2` `{shared.average_score_mean:.6f}`"
        )

    released_headline = ""
    if released_phase_2_winner is not None:
        released_headline = (
            f"The released phase-2 winner among saved checkpoints is "
            f"`{released_phase_2_winner.variant}` at "
            f"`{released_phase_2_winner.average_score_mean:.6f}` across "
            f"`{len(released_phase_2_winner.rows)}` eval jobs."
        )

    shared_phase_3_text = ""
    if shared_phase_3 is not None:
        shared_phase_3_text = (
            f"The shared DTDE checkpoint family also completed the released `phase_3_*` "
            f"six-building sweep at `{shared_phase_3.average_score_mean:.6f}` across "
            f"`{len(shared_phase_3.rows)}` eval jobs."
        )

    ppo_public_rows = [
        row for row in ppo_sweep_summary_rows if row["split"] == "public_dev"
    ]
    ppo_public_rows.sort(key=lambda row: row["lr"])
    ppo_public_text = "\n".join(
        (
            f"- Shared DTDE PPO reward_v2 lr=`{row['lr']}` on `public_dev`: mean "
            f"`{row['average_score_mean']:.6f}`, std `{row['average_score_std']:.6f}`, "
            f"95% CI `{row['average_score_ci95']:.6f}`, seeds `{row['seed_count']}`"
        )
        for row in ppo_public_rows
    )
    ppo_phase3_group_lines: list[str] = []
    for lr in sorted({row.lr for row in ppo_sweep_rows}):
        phase3_metrics = [
            row.metric
            for row in ppo_sweep_rows
            if row.lr == lr and row.metric.split.startswith("phase_3_")
        ]
        if not phase3_metrics:
            continue
        summary = _summarize_rows(phase3_metrics)
        ppo_phase3_group_lines.append(
            f"- Shared DTDE PPO reward_v2 lr=`{lr}` across released `phase_3_*`: mean "
            f"`{summary.average_score_mean:.6f}`, std `{summary.average_score_std:.6f}`, "
            f"95% CI `{summary.average_score_ci95:.6f}`, eval jobs `{len(summary.rows)}`"
        )
    ppo_phase3_text = "\n".join(ppo_phase3_group_lines)
    ppo_public_headline = ""
    if ppo_public_rows:
        best_ppo_public = min(ppo_public_rows, key=lambda row: row["average_score_mean"])
        ppo_public_headline = (
            f"The best shared-PPO local sweep setting is lr=`{best_ppo_public['lr']}` at "
            f"`{best_ppo_public['average_score_mean']:.6f}` on `public_dev`."
        )
    ppo_artifact_caveat = (
        "the centralized PPO baseline artifact is still missing locally, so the repo has "
        "shared-PPO sweep evidence but not the fixed-topology PPO baseline row yet"
    )
    ppo_public_block = (
        ppo_public_text
        if ppo_public_text
        else "- shared PPO sweep artifacts are not present locally yet"
    )
    ppo_phase3_block = (
        ppo_phase3_text
        if ppo_phase3_text
        else "- released phase_3 shared-PPO sweep artifacts are not present locally yet"
    )
    released_phase_3_line = (
        f"- {shared_phase_3_text}"
        if shared_phase_3_text
        else "- released phase_3 shared-checkpoint results are not available yet"
    )
    sac_claim_line = (
        "- the central SAC baseline, central `reward_v1`, and central `reward_v2` "
        "now all have claim-quality 5-seed local comparisons"
    )
    reward_compare_line = (
        f"- `central_reward_v1` currently beats `central_reward_v2` on mean total score "
        f"(`{reward_v1.average_score_mean:.6f}` vs `{reward_v2.average_score_mean:.6f}`)"
    )
    phase2_winner_line = (
        "- `central_reward_v2` is the best saved fixed-topology checkpoint family on "
        "the released phase-2 online-eval datasets"
    )
    released_split_block = chr(10).join(released_split_lines) if released_split_lines else ""
    chesca_caveat = (
        "- some local SAC means are numerically below the published CHESCA "
        "references, but that is **not** enough to claim a true leaderboard win"
    )
    released_dataset_caveat = (
        "- the released phase-2 online-eval datasets are much closer to the official "
        "evaluator-side setting than `public_dev`, but they are still reported "
        "separately from the local tuning split"
    )
    phase3_portability_caveat = (
        "- centralized checkpoints are not portable to the released six-building "
        "`phase_3_*` datasets, so the current `phase_3` evidence is "
        "shared-controller-only"
    )

    text = f"""# Current Results Snapshot

These files are the clean tracked summary of the raw outputs under `results/`.

## Files in this directory

- `local_main_results.csv` — tracked method-level summary rows
- `sac_ablation_summary.csv` — SAC-only variant comparison with seed-aware means and CIs
- `sac_seed_inventory.csv` — per-seed SAC run inventory for the local phase-2 batch
- `released_eval_main_results.csv` — released official-eval family summaries
- `released_eval_seed_inventory.csv` — per-seed released-eval checkpoint inventory
- `ppo_shared_sweep_summary.csv` — per-learning-rate shared-PPO sweep summary rows
- `ppo_shared_sweep_inventory.csv` — per-run shared-PPO sweep inventory with KPI columns
- `official_benchmark_reference.csv` — published CityLearn 2023 reference numbers

## Local public_dev snapshot

- local RBC baseline: `{rbc_score:.6f}`
- PPO baseline artifact: missing locally
{line(by_variant['central_baseline'])}
{line(by_variant['central_reward_v1'])}
{line(by_variant['central_reward_v2'])}
{line(by_variant['shared_dtde_reward_v2'])}

Lower is better.

## PPO shared sweep snapshot

- centralized PPO baseline artifact: still missing locally
{ppo_public_block}
{ppo_public_headline}
{ppo_phase3_block}

## Local tuning headline

{headline_intro}

The strongest improved SAC result with a full 5-seed comparison is
`{best_claim_summary.variant}` at `{best_claim_summary.average_score_mean:.6f}`.
That is `{best_claim_delta:.6f}` lower than the local RBC baseline, a
`{best_claim_pct:.2f}%` improvement on the local phase-2 evaluation dataset.

## Released official-eval snapshot

{released_phase_2_text}
{released_phase_3_line}

Lower is better.

## Current headline

{released_headline}

## What the full SAC evidence says

- every measured SAC variant beats the local RBC baseline by a large margin
{sac_claim_line}
{reward_compare_line}
{phase2_winner_line}
- shared / decentralized `reward_v2` did not beat the released phase-2 central winner
{released_split_block}
- none of the saved checkpoints currently beat the published CHESCA references

## Important caveats

- these numbers are local phase-2 evaluation numbers, not official leaderboard results
{chesca_caveat}
- {ppo_artifact_caveat}
{released_dataset_caveat}
{phase3_portability_caveat}
"""
    (OUTPUT_ROOT / "README.md").write_text(text)


def _missing_canonical_metric_requirements() -> list[str]:
    if not any(METRICS_ROOT.glob("*.csv")):
        return ["no metric CSVs under results/metrics"]

    missing: list[str] = []
    if not list(METRICS_ROOT.glob("rbc__basic_rbc__public_dev__seed0__*.csv")):
        missing.append("rbc__basic_rbc__public_dev__seed0__*.csv")

    local_sac_keys: set[tuple[str, int]] = set()
    released_groups: set[str] = set()
    for path in sorted(METRICS_ROOT.glob("sac__*.csv")):
        row = _read_metric_row(path)
        if row.split == "public_dev":
            local_sac_keys.add((row.variant, row.seed))
        elif row.split.startswith("phase_"):
            released_groups.add(_released_group(row.split))

    for variant in REQUIRED_LOCAL_SAC_VARIANTS:
        seeds = {seed for row_variant, seed in local_sac_keys if row_variant == variant}
        if not seeds:
            missing.append(f"sac__{variant}__public_dev__*.csv")
        elif len(seeds) < 5:
            missing.append(f"sac__{variant}__public_dev__ needs 5 seeds, found {len(seeds)}")

    if "released_phase_2_online_eval" not in released_groups:
        missing.append("SAC released phase_2_online_eval metrics")
    if "released_phase_3" not in released_groups:
        missing.append("SAC released phase_3 metrics")

    sweep_summary_path = REPO_ROOT / "results" / "sweep" / "summary.csv"
    if sweep_summary_path.exists():
        with sweep_summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["algo"] != "ppo":
                    continue
                metric_path = METRICS_ROOT / f"{row['run_id']}.csv"
                if not metric_path.exists():
                    missing.append(f"PPO sweep metric {metric_path.relative_to(REPO_ROOT)}")

    return missing


def _validate_tracked_outputs() -> None:
    missing = [name for name in TRACKED_OUTPUT_FILES if not (OUTPUT_ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(
            "no canonical metrics found and tracked submission outputs are missing: "
            + ", ".join(missing)
        )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    missing_canonical = _missing_canonical_metric_requirements()
    if missing_canonical:
        _validate_tracked_outputs()
        print(
            "Canonical metrics under results/metrics are missing or partial "
            f"({'; '.join(missing_canonical)}); "
            "leaving tracked submission/results outputs unchanged."
        )
        return

    rbc_row, sac_rows = _load_local_rows()
    released_rows = _load_released_rows()
    ppo_sweep_rows = _load_ppo_sweep_rows()
    sac_summaries = _build_sac_summaries(sac_rows)
    released_group_summaries = _build_released_group_summaries(released_rows)
    released_split_summaries = _build_released_split_summaries(released_rows)

    local_rows = _build_local_results_rows(rbc_row, sac_summaries)
    ablation_rows = _build_sac_ablation_rows(rbc_row, sac_summaries)
    seed_rows = _build_seed_inventory_rows(sac_rows)
    released_main_rows = _build_released_main_rows(released_group_summaries)
    released_seed_rows = _build_released_seed_inventory_rows(released_rows)
    ppo_sweep_summary_rows = _build_ppo_sweep_summary_rows(ppo_sweep_rows)
    ppo_sweep_inventory_rows = _build_ppo_sweep_inventory_rows(ppo_sweep_rows)
    reference_rows = _build_reference_rows()

    _write_csv(
        OUTPUT_ROOT / "local_main_results.csv",
        [
            "method_id",
            "method_label",
            "status",
            "evidence_level",
            "algorithm",
            "variant",
            "split",
            "seed_count",
            "average_score_mean",
            "average_score_std",
            "average_score_ci95",
            "best_average_score",
            "worst_average_score",
            "delta_vs_local_rbc_mean",
            "pct_improvement_vs_local_rbc_mean",
            "district_cost_total_mean",
            "district_carbon_emissions_total_mean",
            "district_daily_peak_average_mean",
            "district_discomfort_proportion_mean",
            "district_one_minus_thermal_resilience_proportion_mean",
            "best_run_id",
            "worst_run_id",
            "notes",
        ],
        local_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "sac_ablation_summary.csv",
        [
            "variant",
            "method_label",
            "seed_count",
            "evidence_level",
            "average_score_mean",
            "average_score_std",
            "average_score_ci95",
            "best_average_score",
            "worst_average_score",
            "delta_vs_central_baseline_mean",
            "pct_improvement_vs_central_baseline_mean",
            "delta_vs_local_rbc_mean",
            "district_cost_total_mean",
            "district_carbon_emissions_total_mean",
            "district_daily_peak_average_mean",
            "district_discomfort_proportion_mean",
            "district_one_minus_thermal_resilience_proportion_mean",
        ],
        ablation_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "sac_seed_inventory.csv",
        [
            "run_id",
            "file_name",
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
        ],
        seed_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "released_eval_main_results.csv",
        [
            "method_id",
            "method_label",
            "algorithm",
            "variant",
            "eval_group",
            "split_count",
            "eval_job_count",
            "seed_count",
            "average_score_mean",
            "average_score_std",
            "average_score_ci95",
            "best_average_score",
            "worst_average_score",
            "delta_vs_released_phase2_winner_mean",
            "district_cost_total_mean",
            "district_carbon_emissions_total_mean",
            "district_daily_peak_average_mean",
            "district_discomfort_proportion_mean",
            "district_one_minus_thermal_resilience_proportion_mean",
            "notes",
        ],
        released_main_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "released_eval_seed_inventory.csv",
        [
            "run_id",
            "file_name",
            "algorithm",
            "variant",
            "split",
            "eval_group",
            "seed",
            "dataset_name",
            "average_score",
            "district_cost_total",
            "district_carbon_emissions_total",
            "district_daily_peak_average",
            "district_discomfort_proportion",
            "district_one_minus_thermal_resilience_proportion",
        ],
        released_seed_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "official_benchmark_reference.csv",
        ["method_id", "method_label", "split_type", "average_score", "source_note"],
        reference_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "ppo_shared_sweep_summary.csv",
        [
            "method_id",
            "method_label",
            "algorithm",
            "variant",
            "lr",
            "split",
            "seed_count",
            "evidence_level",
            "average_score_mean",
            "average_score_std",
            "average_score_ci95",
            "best_average_score",
            "worst_average_score",
            "district_cost_total_mean",
            "district_carbon_emissions_total_mean",
            "district_daily_peak_average_mean",
            "district_discomfort_proportion_mean",
            "district_one_minus_thermal_resilience_proportion_mean",
            "best_run_id",
            "worst_run_id",
            "notes",
        ],
        ppo_sweep_summary_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "ppo_shared_sweep_inventory.csv",
        [
            "run_id",
            "file_name",
            "algorithm",
            "variant",
            "lr",
            "split",
            "seed",
            "dataset_name",
            "average_score",
            "district_cost_total",
            "district_carbon_emissions_total",
            "district_daily_peak_average",
            "district_discomfort_proportion",
            "district_one_minus_thermal_resilience_proportion",
        ],
        ppo_sweep_inventory_rows,
    )
    _write_status_markdown(
        rbc_row,
        sac_summaries,
        released_group_summaries,
        released_split_summaries,
        ppo_sweep_rows,
        ppo_sweep_summary_rows,
    )


if __name__ == "__main__":
    main()

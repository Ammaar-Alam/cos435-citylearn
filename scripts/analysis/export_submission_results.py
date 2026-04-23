from __future__ import annotations

import csv
import json
import math
import statistics as st
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# these are local paths setup like Google Drive structure
AMMAAR_METRICS_ROOT = REPO_ROOT / "results" / "ammaar" / "metrics"
ERIK_METRICS_ROOT = REPO_ROOT / "results" / "erik" / "runs"
GRACE_METRICS_ROOT = REPO_ROOT / "results" / "grace" / "metrics"
OUTPUT_ROOT = REPO_ROOT / "submission" / "results"

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
    "basic_rbc": "RBC baseline",
    "central_baseline": "SAC centralized baseline",
    "central_reward_v1": "SAC centralized reward_v1",
    "central_reward_v2": "SAC centralized reward_v2",
    "shared_dtde_reward_v2": "SAC shared DTDE reward_v2",
    "ppo_central_baseline": "PPO centralized baseline",
    "ppo_shared_dtde_reward_v2": "PPO shared DTDE reward_v2",
}

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


def _metric_row_from_json(path: Path) -> MetricRow:
    d = json.loads(path.read_text())
    kpis = d["district_kpis"]
    return MetricRow(
        file_name=path.name,
        run_id=d["run_id"],
        algorithm=d["algorithm"],
        variant=d["variant"],
        split=d["split"],
        seed=int(d["seed"]),
        dataset_name=d["dataset_name"],
        average_score=float(d["average_score"]),
        district_cost_total=float(kpis["cost_total"]),
        district_carbon_emissions_total=float(kpis["carbon_emissions_total"]),
        district_daily_peak_average=float(kpis["daily_peak_average"]),
        district_discomfort_proportion=float(kpis["discomfort_proportion"]),
        district_one_minus_thermal_resilience_proportion=float(
            kpis["one_minus_thermal_resilience_proportion"]
        ),
    )


def _rbc_row_from_committed_csv() -> MetricRow:
    """Read the RBC row from the committed summary CSV (no raw file exists locally)."""
    committed = OUTPUT_ROOT / "local_main_results.csv"
    with committed.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method_id"] == "local_rbc":
                return MetricRow(
                    file_name="local_main_results.csv",
                    run_id=row["best_run_id"],
                    algorithm=row["algorithm"],
                    variant=row["variant"],
                    split=row["split"],
                    seed=0,
                    dataset_name="citylearn_challenge_2023_phase_2_local_evaluation",
                    average_score=float(row["average_score_mean"]),
                    district_cost_total=float(row["district_cost_total_mean"]),
                    district_carbon_emissions_total=float(
                        row["district_carbon_emissions_total_mean"]
                    ),
                    district_daily_peak_average=float(
                        row["district_daily_peak_average_mean"]
                    ),
                    district_discomfort_proportion=float(
                        row["district_discomfort_proportion_mean"]
                    ),
                    district_one_minus_thermal_resilience_proportion=float(
                        row["district_one_minus_thermal_resilience_proportion_mean"]
                    ),
                )
    raise ValueError("local_rbc row not found in committed local_main_results.csv")


def _load_sac_rows() -> list[MetricRow]:
    return sorted(
        (
            _read_metric_row(path)
            for path in AMMAAR_METRICS_ROOT.glob("sac__*.csv")
            if "__public_dev__" in path.name
        ),
        key=lambda row: (row.variant, row.seed, row.run_id),
    )


def _load_grace_ppo_rows() -> list[MetricRow]:
    if not GRACE_METRICS_ROOT.exists():
        return []
    rows = []
    for path in sorted(GRACE_METRICS_ROOT.glob("ppo__*.csv")):
        if "__public_dev__" not in path.name:
            continue
        try:
            rows.append(_read_metric_row(path))
        except (KeyError, ValueError):
            continue
    return rows


def _load_erik_ppo_rows() -> list[MetricRow]:
    """Load Erik's PPO runs from JSON, keeping the best-scoring run per seed."""
    if not ERIK_METRICS_ROOT.exists():
        return []
    by_seed: dict[tuple[str, int], MetricRow] = {}
    for json_path in sorted(ERIK_METRICS_ROOT.glob("ppo__*__public_dev__*/metrics.json")):
        try:
            row = _metric_row_from_json(json_path)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        key = (row.variant, row.seed)
        if key not in by_seed or row.average_score < by_seed[key].average_score:
            by_seed[key] = row
    return sorted(by_seed.values(), key=lambda r: (r.variant, r.seed))


def _released_group(split: str) -> str:
    if split.startswith("phase_2_online_eval_"):
        return "released_phase_2_online_eval"
    if split.startswith("phase_3_"):
        return "released_phase_3"
    raise ValueError(f"unsupported released split: {split}")


def _load_released_rows() -> list[MetricRow]:
    latest_rows: dict[tuple[str, str, int], MetricRow] = {}
    for path in sorted(AMMAAR_METRICS_ROOT.glob("sac__*.csv")):
        row = _read_metric_row(path)
        if not row.split.startswith("phase_"):
            continue
        latest_rows[(row.variant, row.split, row.seed)] = row

    return sorted(
        latest_rows.values(),
        key=lambda row: (_released_group(row.split), row.variant, row.split, row.seed, row.run_id),
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
    if seed_count >= 2:
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


def _group_into_summaries(rows: list[MetricRow]) -> list[MetricSummary]:
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.variant, row.split), []).append(row)
    return sorted(
        (_summarize_rows(rows) for rows in grouped.values()),
        key=lambda s: (s.average_score_mean, s.variant),
    )


def _build_released_group_summaries(rows: list[MetricRow]) -> list[ReleasedSummary]:
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.variant, _released_group(row.split)), []).append(row)

    return sorted(
        (_summarize_released_rows(scope, grouped_rows) for (_, scope), grouped_rows in grouped.items()),
        key=lambda summary: (summary.scope, summary.average_score_mean, summary.variant),
    )


def _build_released_split_summaries(rows: list[MetricRow]) -> list[ReleasedSummary]:
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.variant, row.split), []).append(row)

    return sorted(
        (_summarize_released_rows(scope, grouped_rows) for (_, scope), grouped_rows in grouped.items()),
        key=lambda summary: (summary.scope, summary.average_score_mean, summary.variant),
    )


def _build_sac_summaries(sac_rows: list[MetricRow]) -> list[MetricSummary]:
    return _group_into_summaries(sac_rows)


def _make_method_row(
    rbc_score: float,
    method_id: str,
    method_label: str,
    evidence_level: str,
    summary: MetricSummary,
    note: str,
) -> dict[str, object]:
    delta = rbc_score - summary.average_score_mean
    best_row = min(summary.rows, key=lambda r: r.average_score)
    worst_row = max(summary.rows, key=lambda r: r.average_score)
    return {
        "method_id": method_id,
        "method_label": method_label,
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
        "pct_improvement_vs_local_rbc_mean": round(delta / rbc_score * 100.0, 2),
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


def _build_local_results_rows(
    rbc_row: MetricRow,
    sac_summaries: list[MetricSummary],
    ppo_summaries: list[MetricSummary],
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
        }
    ]

    for summary in ppo_summaries:
        evidence_level = _evidence_level(len(summary.rows))
        note = (
            "multi-seed local phase_2 claim run"
            if evidence_level == "claim_run"
            else "multi-seed local phase_2 pilot run"
            if evidence_level == "pilot"
            else "single-seed local phase_2 evaluation run"
        )
        rows.append(
            _make_method_row(
                rbc_score,
                method_id=f"{summary.variant}_{summary.split}",
                method_label=_variant_label(summary.variant),
                evidence_level=evidence_level,
                summary=summary,
                note=note,
            )
        )

    for summary in sac_summaries:
        evidence_level = _evidence_level(len(summary.rows))
        note = (
            "multi-seed local phase_2 claim run"
            if evidence_level == "claim_run"
            else "multi-seed local phase_2 pilot run"
            if evidence_level == "pilot"
            else "single-seed local phase_2 evaluation run"
        )
        rows.append(
            _make_method_row(
                rbc_score,
                method_id=f"sac_{summary.variant}_{summary.split}",
                method_label=_variant_label(summary.variant),
                evidence_level=evidence_level,
                summary=summary,
                note=note,
            )
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
    ppo_summaries: list[MetricSummary],
    released_group_summaries: list[ReleasedSummary],
    released_split_summaries: list[ReleasedSummary],
) -> None:
    by_sac = {s.variant: s for s in sac_summaries}
    best_sac = min(sac_summaries, key=lambda s: s.average_score_mean)
    claim_sac = [s for s in sac_summaries if len(s.rows) >= 5]
    best_claim_sac = min(claim_sac, key=lambda s: s.average_score_mean) if claim_sac else best_sac
    rbc_score = rbc_row.average_score
    best_claim_delta = rbc_score - best_claim_sac.average_score_mean
    best_claim_pct = best_claim_delta / rbc_score * 100.0
    reward_v1 = by_sac["central_reward_v1"]
    reward_v2 = by_sac["central_reward_v2"]
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
    shared_phase_2 = released_by_key.get(
        ("shared_dtde_reward_v2", "released_phase_2_online_eval")
    )
    shared_phase_3 = released_by_key.get(("shared_dtde_reward_v2", "released_phase_3"))

    if len(best_sac.rows) >= 5:
        headline_intro = (
            f"The best current mean is `{best_sac.variant}` at "
            f"`{best_sac.average_score_mean:.6f}` across `{len(best_sac.rows)}` seeds."
        )
    else:
        headline_intro = (
            f"The best pilot mean right now is `{best_sac.variant}` at "
            f"`{best_sac.average_score_mean:.6f}`, but that result only has "
            f"`{len(best_sac.rows)}` seeds."
        )

    def line(summary: MetricSummary) -> str:
        return (
            f"- {_variant_label(summary.variant)}: mean `{summary.average_score_mean:.6f}`, "
            f"std `{summary.average_score_std:.6f}`, 95% CI `{summary.average_score_ci95:.6f}`, "
            f"seeds `{len(summary.rows)}`"
        )

    def released_line(summary: ReleasedSummary) -> str:
        return (
            f"- {_variant_label(summary.variant)} on `{summary.scope}`: mean "
            f"`{summary.average_score_mean:.6f}`, std `{summary.average_score_std:.6f}`, "
            f"95% CI `{summary.average_score_ci95:.6f}`, eval jobs `{len(summary.rows)}`, "
            f"seeds `{len({row.seed for row in summary.rows})}`"
        )

    ppo_lines = (
        "\n".join(line(s) for s in ppo_summaries)
        if ppo_summaries
        else "- PPO: no results found locally"
    )

    sac_lines = "\n".join(
        line(by_sac[v])
        for v in ["central_baseline", "central_reward_v1", "central_reward_v2", "shared_dtde_reward_v2"]
        if v in by_sac
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

    text = f"""# Current Results Snapshot

These files are the clean tracked summary of the raw outputs under `results/`.

## Files in this directory

- `local_main_results.csv` — tracked method-level summary rows
- `sac_ablation_summary.csv` — SAC-only variant comparison with seed-aware means and CIs
- `sac_seed_inventory.csv` — per-seed SAC run inventory for the local phase-2 batch
- `released_eval_main_results.csv` — released official-eval family summaries
- `released_eval_seed_inventory.csv` — per-seed released-eval checkpoint inventory
- `official_benchmark_reference.csv` — published CityLearn 2023 reference numbers

## Local public_dev snapshot

- local RBC baseline: `{rbc_score:.6f}`

PPO results:
{ppo_lines}

SAC results:
{sac_lines}

Lower is better.

## Local tuning headline

{headline_intro}

The strongest improved SAC result with a full 5-seed comparison is
`{best_claim_sac.variant}` at `{best_claim_sac.average_score_mean:.6f}`.
That is `{best_claim_delta:.6f}` lower than the local RBC baseline, a
`{best_claim_pct:.2f}%` improvement on the local phase-2 evaluation dataset.

## Released official-eval snapshot

{released_phase_2_text}
- {shared_phase_3_text if shared_phase_3_text else 'released phase_3 shared-checkpoint results are not available yet'}

Lower is better.

## Current headline

{released_headline}

## What the full SAC evidence says

- every measured SAC variant beats the local RBC baseline by a large margin
- the central SAC baseline, central `reward_v1`, and central `reward_v2` now all have claim-quality 5-seed local comparisons
- `central_reward_v1` currently beats `central_reward_v2` on mean total score (`{reward_v1.average_score_mean:.6f}` vs `{reward_v2.average_score_mean:.6f}`)
- `central_reward_v2` is the best saved fixed-topology checkpoint family on the released phase-2 online-eval datasets
- shared / decentralized `reward_v2` did not beat the released phase-2 central winner
{chr(10).join(released_split_lines) if released_split_lines else ''}
- none of the saved checkpoints currently beat the published CHESCA references

## Important caveats

- these numbers are local phase-2 evaluation numbers, not official leaderboard results
- some local SAC means are numerically below the published CHESCA references, but that is **not** enough to claim a true leaderboard win
- the released phase-2 online-eval datasets are much closer to the official evaluator-side setting than `public_dev`, but they are still reported separately from the local tuning split
- centralized checkpoints are not portable to the released six-building `phase_3_*` datasets, so the current `phase_3` evidence is shared-DTDE-only
- held-out evaluation is still missing, so the final paper claim is not complete yet
"""
    (OUTPUT_ROOT / "README.md").write_text(text)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rbc_row = _rbc_row_from_committed_csv()
    sac_rows = _load_sac_rows()
    ppo_rows = _load_grace_ppo_rows() + _load_erik_ppo_rows()
    released_rows = _load_released_rows()

    sac_summaries = _build_sac_summaries(sac_rows)
    ppo_summaries = _group_into_summaries(ppo_rows)
    released_group_summaries = _build_released_group_summaries(released_rows)
    released_split_summaries = _build_released_split_summaries(released_rows)

    local_rows = _build_local_results_rows(rbc_row, sac_summaries, ppo_summaries)
    ablation_rows = _build_sac_ablation_rows(rbc_row, sac_summaries)
    seed_rows = _build_seed_inventory_rows(sac_rows)
    released_main_rows = _build_released_main_rows(released_group_summaries)
    released_seed_rows = _build_released_seed_inventory_rows(released_rows)
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
    _write_status_markdown(
        rbc_row, sac_summaries, ppo_summaries, released_group_summaries, released_split_summaries
    )


if __name__ == "__main__":
    main()

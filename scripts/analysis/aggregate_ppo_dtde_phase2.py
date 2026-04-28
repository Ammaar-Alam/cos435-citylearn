"""Aggregate PPO-DTDE reward_v2 per-split held-out results.

Pulls PPO DTDE reward_v2 eval scores off local run artifacts and writes
per-split means into submission/results/cross_split_scores.csv for the
'PPO DTDE' row. Two data sources:

  * phase_2_online_eval_{1,2,3}: parsed from the latest
    results/eval_phase2_ppo_dtde_*/progress.log. Selects the best-public_dev
    learning-rate checkpoint per seed (one per seed) before averaging
    (eval_job_count = 10 seeds * 3 splits).
  * phase_3_{1,2,3}: averaged across results/runs/
    ppo__ppo_shared_dtde_reward_v2__phase_3_* metrics.json files, filtered
    to the same best-public_dev LR per seed via manifest.artifact_id so
    cross_split_scores.csv stays consistent with the n=30 phase_3 row already
    reported in released_eval_main_results.csv.

This script also backfills the PPO DTDE reward_v2 phase_2 row into
submission/results/released_eval_main_results.csv (full stats: mean/std/ci95
+ KPI means) and per-eval rows into released_eval_seed_inventory.csv, so the
released-eval snapshot stays consistent with cross_split_scores.csv.

Usage:
    .venv/bin/python scripts/analysis/aggregate_ppo_dtde_phase2.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "results" / "runs"
EVAL_DIRS_GLOB = "eval_phase2_ppo_dtde_*"
CROSS_SPLIT_CSV = REPO_ROOT / "submission" / "results" / "cross_split_scores.csv"
RELEASED_MAIN_CSV = REPO_ROOT / "submission" / "results" / "released_eval_main_results.csv"
RELEASED_INVENTORY_CSV = REPO_ROOT / "submission" / "results" / "released_eval_seed_inventory.csv"
PUBLIC_DEV_PREFIX = "ppo__ppo_shared_dtde_reward_v2__public_dev__"

PROGRESS_OK = re.compile(
    r"eval\s+(?P<run_id>\S+)\s+on\s+(?P<split>phase_2_online_eval_[123])\s+\(seed=(?P<seed>\d+)\)"
)
PROGRESS_SCORE = re.compile(r"OK\s+score=([0-9.]+)")

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


def _std(values: list[float]) -> float:
    return st.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    critical = T_CRITICAL_95.get(len(values), 1.96)
    return critical * _std(values) / math.sqrt(len(values))


def _require_complete_split_coverage(
    by_split: dict[str, list[float]],
    expected_splits: list[str],
    expected_count: int,
    *,
    label: str,
) -> None:
    missing_or_partial = {
        split: len(by_split.get(split, []))
        for split in expected_splits
        if len(by_split.get(split, [])) != expected_count
    }
    if missing_or_partial:
        details = ", ".join(
            f"{split}={count}/{expected_count}"
            for split, count in sorted(missing_or_partial.items())
        )
        raise SystemExit(f"incomplete {label} coverage; refusing to write summaries: {details}")


def _require_complete_metric_coverage(
    metrics: list[dict],
    expected_splits: list[str],
    expected_seeds: set[int],
    *,
    label: str,
) -> None:
    seen: set[tuple[str, int]] = set()
    duplicates: list[tuple[str, int]] = []
    for metric in metrics:
        key = (str(metric.get("split", "")), int(metric.get("seed", -1)))
        if key in seen:
            duplicates.append(key)
        seen.add(key)

    expected = {
        (split, seed)
        for split in expected_splits
        for seed in expected_seeds
    }
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected or duplicates:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(f"{split}/seed{seed}" for split, seed in missing))
        if unexpected:
            parts.append(
                "unexpected " + ", ".join(f"{split}/seed{seed}" for split, seed in unexpected)
            )
        if duplicates:
            parts.append(
                "duplicate " + ", ".join(f"{split}/seed{seed}" for split, seed in duplicates)
            )
        details = "; ".join(parts)
        raise SystemExit(
            f"incomplete {label} coverage; refusing to write summaries: {details}"
        )


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_public_dev_scores() -> dict[int, list[tuple[str, float]]]:
    """Return {seed: [(run_id, public_dev_score), ...]} for PPO DTDE reward_v2."""
    by_seed: dict[int, list[tuple[str, float]]] = {}
    for run_dir in sorted((REPO_ROOT / "results" / "runs").glob(f"{PUBLIC_DEV_PREFIX}seed*")):
        m = re.search(r"seed(\d+)__", run_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        d = json.loads(metrics_path.read_text())
        score = d.get("average_score")
        if score is None:
            continue
        by_seed.setdefault(seed, []).append((run_dir.name, float(score)))
    return by_seed


def _parse_progress_log(log_path: Path) -> list[tuple[str, str, int, float]]:
    """Extract (run_id, split, seed, score) tuples from a progress.log file."""
    out: list[tuple[str, str, int, float]] = []
    pending: tuple[str, str, int] | None = None
    for line in log_path.read_text().splitlines():
        m_eval = PROGRESS_OK.search(line)
        if m_eval:
            pending = (m_eval["run_id"], m_eval["split"], int(m_eval["seed"]))
            continue
        if pending is not None:
            m_score = PROGRESS_SCORE.search(line)
            if m_score:
                out.append((*pending, float(m_score.group(1))))
                pending = None
            elif "FAILED" in line:
                pending = None
    return out


def _load_phase2_eval_metrics_by_artifact(
    best_run_ids: set[str],
) -> list[dict]:
    """Scan results/runs/ for PPO DTDE phase_2 eval runs whose manifest
    artifact_id is in ``best_run_ids`` and return their full metrics dicts.

    Each returned dict extends the per-run metrics.json with an `artifact_id`
    key so callers can join back to the selected training checkpoint.
    """
    out: list[dict] = []
    for run_dir in sorted(RUNS_DIR.glob("ppo__ppo_shared_dtde_reward_v2__phase_2_online_eval_*")):
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        if not manifest_path.exists() or not metrics_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        artifact_id = manifest.get("artifact_id")
        if artifact_id not in best_run_ids:
            continue
        metrics = json.loads(metrics_path.read_text())
        metrics["artifact_id"] = artifact_id
        metrics["run_dir_name"] = run_dir.name
        out.append(metrics)
    return out


def _aggregate_released_phase2_row(
    phase2_metrics: list[dict],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Return (main_csv_row, inventory_rows) for PPO DTDE phase_2 on the
    released_phase_2_online_eval eval group.

    main_csv_row matches the schema of
    submission/results/released_eval_main_results.csv.
    inventory_rows match submission/results/released_eval_seed_inventory.csv.
    """
    if not phase2_metrics:
        raise SystemExit("no phase_2 PPO DTDE metrics.json files matched best-LR set")

    scores = [float(m["average_score"]) for m in phase2_metrics]
    costs = [float(m["district_kpis"]["cost_total"]) for m in phase2_metrics]
    carbons = [float(m["district_kpis"]["carbon_emissions_total"]) for m in phase2_metrics]
    peaks = [float(m["district_kpis"]["daily_peak_average"]) for m in phase2_metrics]
    discomforts = [float(m["district_kpis"]["discomfort_proportion"]) for m in phase2_metrics]
    resiliences = [
        float(m["district_kpis"]["one_minus_thermal_resilience_proportion"])
        for m in phase2_metrics
    ]

    splits = sorted({m["split"] for m in phase2_metrics})
    seeds = sorted({int(m["seed"]) for m in phase2_metrics})

    mean = st.fmean(scores)
    main_row: dict[str, str] = {
        "method_id": "ppo_shared_dtde_reward_v2_released_phase_2_online_eval",
        "method_label": "PPO shared DTDE reward_v2",
        "algorithm": "ppo",
        "variant": "ppo_shared_dtde_reward_v2",
        "eval_group": "released_phase_2_online_eval",
        "split_count": str(len(splits)),
        "eval_job_count": str(len(phase2_metrics)),
        "seed_count": str(len(seeds)),
        "average_score_mean": f"{mean:.6f}",
        "average_score_std": f"{_std(scores):.6f}",
        "average_score_ci95": f"{_ci95(scores):.6f}",
        "best_average_score": f"{min(scores):.6f}",
        "worst_average_score": f"{max(scores):.6f}",
        # Filled in by caller once the released phase-2 winner is known.
        "delta_vs_released_phase2_winner_mean": "",
        "district_cost_total_mean": f"{st.fmean(costs):.6f}",
        "district_carbon_emissions_total_mean": f"{st.fmean(carbons):.6f}",
        "district_daily_peak_average_mean": f"{st.fmean(peaks):.6f}",
        "district_discomfort_proportion_mean": f"{st.fmean(discomforts):.6f}",
        "district_one_minus_thermal_resilience_proportion_mean": f"{st.fmean(resiliences):.6f}",
        "notes": (
            "PPO DTDE sweep checkpoint family evaluated on released "
            "phase_2 splits (best-public_dev LR per seed)"
        ),
    }

    inventory_rows: list[dict[str, str]] = []
    for m in sorted(phase2_metrics, key=lambda d: (d["split"], int(d["seed"]), d["run_id"])):
        inventory_rows.append({
            "run_id": m["run_id"],
            # Seed inventory uses .csv sidecars; keep the convention.
            "file_name": f"{m['run_dir_name']}.csv",
            "algorithm": m["algorithm"],
            "variant": m["variant"],
            "split": m["split"],
            "eval_group": "released_phase_2_online_eval",
            "seed": str(m["seed"]),
            "dataset_name": m["dataset_name"],
            "average_score": f"{float(m['average_score']):.6f}",
            "district_cost_total": f"{float(m['district_kpis']['cost_total']):.6f}",
            "district_carbon_emissions_total": (
                f"{float(m['district_kpis']['carbon_emissions_total']):.6f}"
            ),
            "district_daily_peak_average": f"{float(m['district_kpis']['daily_peak_average']):.6f}",
            "district_discomfort_proportion": (
                f"{float(m['district_kpis']['discomfort_proportion']):.6f}"
            ),
            "district_one_minus_thermal_resilience_proportion": (
                f"{float(m['district_kpis']['one_minus_thermal_resilience_proportion']):.6f}"
            ),
        })

    return main_row, inventory_rows


def _patch_released_main_csv(new_row: dict[str, str]) -> None:
    """Add or overwrite the PPO DTDE reward_v2 phase_2 row in
    released_eval_main_results.csv, recomputing delta_vs_released_phase2_winner_mean.
    """
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    found = False
    with RELEASED_MAIN_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row.get("method_id") == new_row["method_id"]:
                rows.append(dict(new_row))
                found = True
            else:
                rows.append(row)
    if not found:
        rows.append(dict(new_row))

    # Recompute delta_vs_released_phase2_winner_mean across all phase_2 rows.
    phase2 = [r for r in rows if r.get("eval_group") == "released_phase_2_online_eval"]
    try:
        winner_mean = min(
            float(r["average_score_mean"])
            for r in phase2
            if r.get("average_score_mean")
        )
    except ValueError:
        winner_mean = None
    if winner_mean is not None:
        for r in phase2:
            if r.get("average_score_mean"):
                delta = float(r["average_score_mean"]) - winner_mean
                r["delta_vs_released_phase2_winner_mean"] = f"{delta:.6f}"

    # Preserve original row order (with the PPO DTDE row slotted where the
    # old PPO phase_3 row used to appear — but since we don't have a stable
    # group ordering beyond insertion, just keep insertion order).
    _write_csv_rows(RELEASED_MAIN_CSV, fieldnames, rows)
    print(f"  updated {RELEASED_MAIN_CSV.relative_to(REPO_ROOT)}")


def _patch_released_inventory_csv(new_rows: list[dict[str, str]]) -> None:
    """Replace PPO DTDE reward_v2 phase_2 rows in released_eval_seed_inventory.csv.

    Rerun batches get new timestamped run_ids, so dropping only matching run_ids
    would preserve stale rows from earlier batches and double-count them in
    downstream cross-split means.
    """
    existing: list[dict[str, str]] = []
    fieldnames: list[str] = []
    drop_keys = {
        (r["algorithm"], r["variant"], r["eval_group"])
        for r in new_rows
    }
    with RELEASED_INVENTORY_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row_key = (row.get("algorithm", ""), row.get("variant", ""), row.get("eval_group", ""))
            if row_key in drop_keys:
                continue
            existing.append(row)

    # Filter new_rows to the columns that exist in the target CSV so we don't
    # accidentally introduce new columns.
    projected = [{f: row.get(f, "") for f in fieldnames} for row in new_rows]
    merged = existing + projected
    merged.sort(
        key=lambda r: (
            r.get("variant", ""),
            r.get("split", ""),
            int(r.get("seed", 0) or 0),
            r.get("run_id", ""),
        )
    )

    _write_csv_rows(RELEASED_INVENTORY_CSV, fieldnames, merged)
    print(f"  updated {RELEASED_INVENTORY_CSV.relative_to(REPO_ROOT)}")


def _load_phase3_per_split_means(
    best_run_ids: set[str] | None = None,
) -> dict[str, float]:
    """Return {p3_1|p3_2|p3_3: mean_score} for PPO DTDE reward_v2.

    If ``best_run_ids`` is given, restrict to eval runs whose manifest
    ``artifact_id`` is in that set (the best-public_dev LR per seed). This
    matches the selection used for released_eval_main_results.csv and keeps
    cross_split_scores.csv internally consistent with the released snapshot.
    Duplicate (split, seed) eval artifacts — accidentally re-run checkpoints
    — are deduped by keeping the first run directory encountered (sorted by
    name). NOTE: within the best-LR-per-seed set, the only duplicates we
    observe are two (phase_3_2 seed9, phase_3_3 seed9) pairs that share the
    same checkpoint and produce identical scores, so dedupe order does not
    affect the reported mean. If ``best_run_ids`` is None, average across all
    phase_3 eval runs (legacy behaviour, kept so the helper works for ad-hoc
    queries).
    """
    from collections import defaultdict
    # Map (split, seed) -> first score seen, to dedupe re-run eval artifacts
    # that share an artifact_id with identical scores.
    seen: dict[tuple[str, int], float] = {}
    for r in sorted(RUNS_DIR.glob("ppo__ppo_shared_dtde_reward_v2__phase_3_*")):
        mpath = r / "metrics.json"
        if not mpath.exists():
            continue
        if best_run_ids is not None:
            manifest_path = r / "manifest.json"
            if not manifest_path.exists():
                continue
            artifact_id = json.loads(manifest_path.read_text()).get("artifact_id")
            if artifact_id not in best_run_ids:
                continue
        d = json.loads(mpath.read_text())
        split = d.get("split")
        score = d.get("average_score")
        seed = d.get("seed")
        if split is None or score is None or seed is None:
            continue
        key = (split, int(seed))
        if key in seen:
            continue  # keep first occurrence (sorted by name ~= oldest timestamp)
        seen[key] = float(score)

    by_split: dict[str, list[float]] = defaultdict(list)
    for (split, _seed), score in seen.items():
        by_split[split].append(score)

    label = {"phase_3_1": "p3_1", "phase_3_2": "p3_2", "phase_3_3": "p3_3"}
    return {
        label[s]: sum(v) / len(v)
        for s, v in by_split.items()
        if s in label
    }


# Cross-split row identity for every tracked method. Each entry maps a display
# label used in cross_split_scores.csv to (algorithm, variant) keys used in
# released_eval_seed_inventory.csv and to the method_id used for the public_dev
# column in local_main_results.csv.
CROSS_SPLIT_METHOD_KEYS: list[tuple[str, str, str, str | None]] = [
    ("RBC", "rbc", "basic_rbc", "local_rbc"),
    ("PPO Central", "ppo", "ppo_central_baseline", "ppo_central_baseline_public_dev"),
    ("PPO DTDE", "ppo", "ppo_shared_dtde_reward_v2", "ppo_shared_dtde_reward_v2_public_dev"),
    ("SAC Central", "sac", "central_baseline", "sac_central_baseline_public_dev"),
    ("SAC rv1", "sac", "central_reward_v1", "sac_central_reward_v1_public_dev"),
    ("SAC rv2", "sac", "central_reward_v2", "sac_central_reward_v2_public_dev"),
    ("SAC DTDE", "sac", "shared_dtde_reward_v2", "sac_shared_dtde_reward_v2_public_dev"),
]

SPLIT_TO_CROSS_COL = {
    "phase_2_online_eval_1": "p2_eval1",
    "phase_2_online_eval_2": "p2_eval2",
    "phase_2_online_eval_3": "p2_eval3",
    "phase_3_1": "p3_1",
    "phase_3_2": "p3_2",
    "phase_3_3": "p3_3",
}

LOCAL_MAIN_CSV = REPO_ROOT / "submission" / "results" / "local_main_results.csv"


def _rebuild_cross_split_from_inventory() -> None:
    """Regenerate every row of cross_split_scores.csv from the authoritative
    sources: submission/results/released_eval_seed_inventory.csv (per-split
    means) and submission/results/local_main_results.csv (public_dev column).

    Preserves the existing fieldnames / row order of cross_split_scores.csv.
    Any method already present in the file but not listed in
    ``CROSS_SPLIT_METHOD_KEYS`` is left untouched.
    """
    if not RELEASED_INVENTORY_CSV.exists() or not LOCAL_MAIN_CSV.exists():
        print("  skip cross_split rebuild: inventory or local_main csv missing")
        return

    inv_rows = list(csv.DictReader(RELEASED_INVENTORY_CSV.open(newline="")))
    local_rows = {r["method_id"]: r for r in csv.DictReader(LOCAL_MAIN_CSV.open(newline=""))}

    # Group inventory by (algorithm, variant, split) -> list of scores.
    from collections import defaultdict
    by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in inv_rows:
        try:
            score = float(r["average_score"])
        except (KeyError, ValueError):
            continue
        by_key[(r["algorithm"], r["variant"], r["split"])].append(score)

    def _per_split_mean(algo: str, variant: str, split: str) -> str:
        vals = by_key.get((algo, variant, split), [])
        return f"{sum(vals) / len(vals):.4f}" if vals else ""

    # Walk the existing CSV to preserve fieldnames + ordering.
    with CROSS_SPLIT_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        existing = list(reader)

    by_method = {r["method"].strip(): r for r in existing}

    for label, algo, variant, local_method_id in CROSS_SPLIT_METHOD_KEYS:
        row = by_method.get(label)
        if row is None:
            # Row absent from CSV — skip rather than invent new rows.
            continue
        # public_dev from local_main_results.csv. Always overwrite with either
        # the current mean or blank so stale values cannot survive rebuilds.
        mean = local_rows.get(local_method_id, {}).get("average_score_mean")
        row["public_dev"] = f"{float(mean):.4f}" if mean else ""
        # per-split columns from inventory.
        for split, col in SPLIT_TO_CROSS_COL.items():
            if col not in fieldnames:
                continue
            row[col] = _per_split_mean(algo, variant, split)

    _write_csv_rows(CROSS_SPLIT_CSV, fieldnames, existing)
    print(f"  rebuilt {CROSS_SPLIT_CSV.relative_to(REPO_ROOT)} from inventory")


def main() -> None:
    # 1) Pick the latest eval_phase2_ppo_dtde_* batch.
    eval_batches = sorted((REPO_ROOT / "results").glob(EVAL_DIRS_GLOB))
    if not eval_batches:
        raise SystemExit(f"no {EVAL_DIRS_GLOB} directories under results/")
    latest = eval_batches[-1]
    log_path = latest / "progress.log"
    print(f"aggregating from: {log_path.relative_to(REPO_ROOT)}")

    eval_rows = _parse_progress_log(log_path)
    print(f"  parsed {len(eval_rows)} successful eval rows from progress.log")
    if not eval_rows:
        raise SystemExit("no successful eval rows found — is the eval still running?")

    # 2) Select best-public_dev LR per seed (lowest public_dev score wins).
    public = _load_public_dev_scores()
    best_per_seed = {
        seed: min(items, key=lambda t: t[1])  # (run_id, score)
        for seed, items in public.items()
    }
    best_run_ids = {rid for rid, _ in best_per_seed.values()}
    print(f"  selected {len(best_run_ids)} best-public_dev checkpoints across {len(public)} seeds")
    if not best_run_ids:
        raise SystemExit("no best-public_dev checkpoints found; cannot aggregate PPO DTDE results")

    # 3) Filter eval rows to only the selected checkpoints, group by split.
    by_split: dict[str, list[float]] = {}
    for run_id, split, _seed, score in eval_rows:
        if run_id in best_run_ids:
            by_split.setdefault(split, []).append(score)

    split_label_map = {
        "phase_2_online_eval_1": "p2_eval1",
        "phase_2_online_eval_2": "p2_eval2",
        "phase_2_online_eval_3": "p2_eval3",
    }
    _require_complete_split_coverage(
        by_split,
        list(split_label_map),
        len(best_run_ids),
        label="phase_2 PPO DTDE eval",
    )

    new_row: dict[str, str] = {"method": "PPO DTDE"}
    print("\n  phase_2 per-split results (best-lr-per-seed):")
    for split, col in split_label_map.items():
        scores = by_split.get(split, [])
        if not scores:
            print(f"    {split}: no data")
            continue
        mean = sum(scores) / len(scores)
        new_row[col] = f"{mean:.4f}"
        print(f"    {split}: mean={mean:.4f} n={len(scores)}")

    # 3b) Phase_3 per-split means from raw run metrics, filtered to the same
    #     best-public_dev LR per seed used above so cross_split_scores.csv
    #     matches released_eval_main_results.csv (n=30 = 10 seeds * 3 splits).
    phase3 = _load_phase3_per_split_means(best_run_ids=best_run_ids)
    if phase3:
        print("\n  phase_3 per-split results (best-lr-per-seed):")
        for col in ("p3_1", "p3_2", "p3_3"):
            if col in phase3:
                new_row[col] = f"{phase3[col]:.4f}"
                print(f"    {col}: mean={phase3[col]:.4f}")

    # 4) Patch cross_split_scores.csv: preserve PPO DTDE public_dev, fill p2_eval*.
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    with CROSS_SPLIT_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row["method"].strip() == "PPO DTDE":
                for col in ("p2_eval1", "p2_eval2", "p2_eval3", "p3_1", "p3_2", "p3_3"):
                    if col in new_row:
                        row[col] = new_row[col]
            rows.append(row)

    _write_csv_rows(CROSS_SPLIT_CSV, fieldnames, rows)
    print(f"\n  updated {CROSS_SPLIT_CSV.relative_to(REPO_ROOT)}")

    # 5) Backfill the released_eval_main_results.csv + seed inventory rows
    #    for PPO DTDE phase_2 with the same best-LR-per-seed selection.
    phase2_metrics = _load_phase2_eval_metrics_by_artifact(best_run_ids)
    expected_phase2_jobs = len(best_run_ids) * len(split_label_map)
    print(
        f"\n  matched {len(phase2_metrics)} phase_2 eval run dirs to best-LR "
        f"checkpoints (expected {expected_phase2_jobs})"
    )
    if len(phase2_metrics) != expected_phase2_jobs:
        raise SystemExit(
            "incomplete phase_2 PPO DTDE metrics coverage; refusing to write summaries: "
            f"{len(phase2_metrics)}/{expected_phase2_jobs}"
        )
    _require_complete_metric_coverage(
        phase2_metrics,
        list(split_label_map),
        set(best_per_seed),
        label="phase_2 PPO DTDE metrics",
    )
    main_row, inventory_rows = _aggregate_released_phase2_row(phase2_metrics)
    print(
        f"  released phase_2: mean={main_row['average_score_mean']} "
        f"std={main_row['average_score_std']} "
        f"ci95={main_row['average_score_ci95']} "
        f"best={main_row['best_average_score']} worst={main_row['worst_average_score']}"
    )
    _patch_released_main_csv(main_row)
    _patch_released_inventory_csv(inventory_rows)

    # 6) Rebuild ALL rows of cross_split_scores.csv from the now-up-to-date
    #    inventory + local_main_results.csv so every method (not just PPO DTDE)
    #    stays consistent with the authoritative per-seed eval data.
    print()
    _rebuild_cross_split_from_inventory()


if __name__ == "__main__":
    main()

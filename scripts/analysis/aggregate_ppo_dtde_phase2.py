"""Aggregate PPO-DTDE reward_v2 per-split held-out results.

Pulls PPO DTDE reward_v2 eval scores off local run artifacts and writes
per-split means into submission/results/cross_split_scores.csv for the
'PPO DTDE' row. Two data sources:

  * phase_2_online_eval_{1,2,3}: parsed from the latest
    results/eval_phase2_ppo_dtde_*/progress.log. Selects the best-public_dev
    learning-rate checkpoint per seed (one per seed) before averaging,
    to match the released phase_3 aggregation methodology (eval_job_count =
    10 seeds * 3 splits).
  * phase_3_{1,2,3}: averaged across all results/runs/
    ppo__ppo_shared_dtde_reward_v2__phase_3_* metrics.json files, matching
    the naive mean already reported in released_eval_main_results.csv.

Usage:
    .venv/bin/python scripts/analysis/aggregate_ppo_dtde_phase2.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "results" / "runs"
EVAL_DIRS_GLOB = "eval_phase2_ppo_dtde_*"
CROSS_SPLIT_CSV = REPO_ROOT / "submission" / "results" / "cross_split_scores.csv"
PUBLIC_DEV_PREFIX = "ppo__ppo_shared_dtde_reward_v2__public_dev__"

PROGRESS_OK = re.compile(
    r"eval\s+(?P<run_id>\S+)\s+on\s+(?P<split>phase_2_online_eval_[123])\s+\(seed=(?P<seed>\d+)\)"
)
PROGRESS_SCORE = re.compile(r"OK\s+score=([0-9.]+)")


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


def _load_phase3_per_split_means() -> dict[str, float]:
    """Return {p3_1|p3_2|p3_3: mean_score} for PPO DTDE reward_v2 across
    all eval runs under results/runs/."""
    from collections import defaultdict
    by_split: dict[str, list[float]] = defaultdict(list)
    for r in sorted(RUNS_DIR.glob("ppo__ppo_shared_dtde_reward_v2__phase_3_*")):
        mpath = r / "metrics.json"
        if not mpath.exists():
            continue
        d = json.loads(mpath.read_text())
        split = d.get("split")
        score = d.get("average_score")
        if split and score is not None:
            by_split[split].append(float(score))
    label = {"phase_3_1": "p3_1", "phase_3_2": "p3_2", "phase_3_3": "p3_3"}
    return {
        label[s]: sum(v) / len(v)
        for s, v in by_split.items()
        if s in label
    }


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

    # 3b) Phase_3 per-split means from raw run metrics.
    phase3 = _load_phase3_per_split_means()
    if phase3:
        print("\n  phase_3 per-split results (all-runs mean):")
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

    CROSS_SPLIT_CSV.write_text(
        ",".join(fieldnames) + "\n"
        + "".join(",".join(row.get(f, "") for f in fieldnames) + "\n" for row in rows)
    )
    print(f"\n  updated {CROSS_SPLIT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""Regenerate cross-split scores from tracked submission result CSVs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MAIN_CSV = REPO_ROOT / "submission" / "results" / "local_main_results.csv"
RELEASED_SEEDS_CSV = REPO_ROOT / "submission" / "results" / "released_eval_seed_inventory.csv"
OUT_CSV = REPO_ROOT / "submission" / "results" / "cross_split_scores.csv"

METHOD_KEYS: list[tuple[str, str, str, str]] = [
    ("RBC", "rbc", "basic_rbc", "local_rbc"),
    ("PPO Central", "ppo", "ppo_central_baseline", "ppo_central_baseline_public_dev"),
    ("PPO DTDE", "ppo", "ppo_shared_dtde_reward_v2", "ppo_shared_dtde_reward_v2_public_dev"),
    ("SAC Central", "sac", "central_baseline", "sac_central_baseline_public_dev"),
    ("SAC rv1", "sac", "central_reward_v1", "sac_central_reward_v1_public_dev"),
    ("SAC rv2", "sac", "central_reward_v2", "sac_central_reward_v2_public_dev"),
    ("SAC DTDE", "sac", "shared_dtde_reward_v2", "sac_shared_dtde_reward_v2_public_dev"),
]

SPLIT_COLUMNS = [
    ("phase_2_online_eval_1", "p2_eval1"),
    ("phase_2_online_eval_2", "p2_eval2"),
    ("phase_2_online_eval_3", "p2_eval3"),
    ("phase_3_1", "p3_1"),
    ("phase_3_2", "p3_2"),
    ("phase_3_3", "p3_3"),
]


def _load_public_dev_scores() -> dict[str, str]:
    with LOCAL_MAIN_CSV.open(newline="") as f:
        return {
            row["method_id"]: f"{float(row['average_score_mean']):.4f}"
            for row in csv.DictReader(f)
            if row.get("average_score_mean")
        }


def _load_released_scores() -> dict[tuple[str, str, str], list[float]]:
    scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with RELEASED_SEEDS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["average_score"])
            except (KeyError, ValueError):
                continue
            key = (row.get("algorithm", ""), row.get("variant", ""), row.get("split", ""))
            scores[key].append(score)
    return scores


def _mean_or_blank(
    scores: dict[tuple[str, str, str], list[float]],
    algorithm: str,
    variant: str,
    split: str,
) -> str:
    values = scores.get((algorithm, variant, split), [])
    return f"{sum(values) / len(values):.4f}" if values else ""


def main() -> None:
    public_dev = _load_public_dev_scores()
    released = _load_released_scores()

    fieldnames = ["method", "public_dev", *(column for _, column in SPLIT_COLUMNS)]
    rows: list[dict[str, str]] = []
    for method, algorithm, variant, public_dev_id in METHOD_KEYS:
        row = {
            "method": method,
            "public_dev": public_dev.get(public_dev_id, ""),
        }
        for split, column in SPLIT_COLUMNS:
            row[column] = _mean_or_blank(released, algorithm, variant, split)
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

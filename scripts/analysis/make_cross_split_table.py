"""Aggregate per-split eval metrics into a cross-split summary table."""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
SUBMISSION_CSV = REPO_ROOT / "submission" / "results" / "local_main_results.csv"
OUT_DIR = REPO_ROOT / "submission" / "tables"

# submission CSV method_id → display label (for public_dev backfill)
SUBMISSION_METHOD_MAP = {
    "local_rbc": "RBC",
    "ppo_central_baseline_public_dev": "PPO Central",
    "ppo_shared_dtde_reward_v2_public_dev": "PPO DTDE",
    "sac_central_baseline_public_dev": "SAC Central",
    "sac_central_reward_v1_public_dev": "SAC rv1",
    "sac_central_reward_v2_public_dev": "SAC rv2",
    "sac_shared_dtde_reward_v2_public_dev": "SAC DTDE",
}

# map run_id prefix → display label
METHOD_LABELS = {
    "rbc__basic_rbc": "RBC",
    "sac__central_baseline": "SAC Central",
    "sac__central_reward_v1": "SAC rv1",
    "sac__central_reward_v2": "SAC rv2",
    "sac__shared_dtde_reward_v2": "SAC DTDE",
    "ppo__ppo_central_baseline": "PPO Central",
    "ppo__ppo_shared_dtde_reward_v2": "PPO DTDE",
}

SPLIT_ORDER = [
    "public_dev",
    "phase_2_online_eval_1",
    "phase_2_online_eval_2",
    "phase_2_online_eval_3",
    "phase_3_1",
    "phase_3_2",
    "phase_3_3",
]

SPLIT_LABELS = {
    "public_dev": "public_dev",
    "phase_2_online_eval_1": "p2_eval1",
    "phase_2_online_eval_2": "p2_eval2",
    "phase_2_online_eval_3": "p2_eval3",
    "phase_3_1": "p3_1",
    "phase_3_2": "p3_2",
    "phase_3_3": "p3_3",
}


def _method_key(run_id: str) -> str | None:
    for prefix, label in METHOD_LABELS.items():
        if run_id.startswith(prefix):
            return label
    return None


def _split_from_run_id(run_id: str) -> str | None:
    for split in SPLIT_ORDER:
        if f"__{split}__" in run_id:
            return split
    return None


def load_metrics() -> dict[tuple[str, str], list[float]]:
    """Returns {(method_label, split): [average_score, ...]}"""
    data: dict[tuple[str, str], list[float]] = {}
    for csv_path in sorted(RESULTS_ROOT.glob("*/metrics/*.csv")):
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                run_id = row.get("run_id", "")
                method = _method_key(run_id)
                split = _split_from_run_id(run_id)
                score_str = row.get("average_score", "")
                if method is None or split is None or not score_str:
                    continue
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                key = (method, split)
                data.setdefault(key, []).append(score)
    return data


def load_submission_public_dev() -> dict[str, float]:
    """Pull public_dev mean scores from the pre-computed submission CSV."""
    scores: dict[str, float] = {}
    if not SUBMISSION_CSV.exists():
        return scores
    with SUBMISSION_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            method_id = row.get("method_id", "")
            label = SUBMISSION_METHOD_MAP.get(method_id)
            score_str = row.get("average_score_mean", "")
            if label and score_str:
                try:
                    scores[label] = float(score_str)
                except ValueError:
                    pass
    return scores


def main() -> None:
    data = load_metrics()

    # backfill public_dev from submission CSV for methods not in local metrics
    for label, score in load_submission_public_dev().items():
        key = (label, "public_dev")
        if key not in data:
            data[key] = [score]

    method_order = ["RBC", "PPO Central", "PPO DTDE", "SAC Central", "SAC rv1", "SAC rv2", "SAC DTDE"]
    methods = [m for m in method_order if any(k[0] == m for k in data)]

    present_splits = [s for s in SPLIT_ORDER if any(k[1] == s for k in data)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cross_split_scores.csv"

    fieldnames = ["method"] + [SPLIT_LABELS[s] for s in present_splits]
    rows = []
    for method in methods:
        row: dict[str, str] = {"method": method}
        for split in present_splits:
            scores = data.get((method, split), [])
            if scores:
                row[SPLIT_LABELS[split]] = f"{sum(scores)/len(scores):.4f}"
            else:
                row[SPLIT_LABELS[split]] = ""
        rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    print()

    col_w = max(len(m) for m in methods)
    header = f"{'method':<{col_w}}" + "".join(f"  {SPLIT_LABELS[s]:>10}" for s in present_splits)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['method']:<{col_w}}"
        for split in present_splits:
            val = row.get(SPLIT_LABELS[split], "")
            line += f"  {val:>10}"
        print(line)


if __name__ == "__main__":
    main()

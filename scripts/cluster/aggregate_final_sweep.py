from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EVAL_SPLITS = (
    "phase_2_online_eval_1",
    "phase_2_online_eval_2",
    "phase_2_online_eval_3",
    "phase_3_1",
    "phase_3_2",
    "phase_3_3",
)


def _label_value(value: str) -> str:
    return value.replace("-", "m").replace(".", "p")


def _expected_cells() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    seeds = (0, 1, 2)
    for lr in ("1e-4", "2e-4", "5e-4", "1e-3"):
        for reward_scaling in ("2.5", "5.0", "10.0"):
            for seed in seeds:
                rows.append(
                    {
                        "algo": "sac",
                        "lr": lr,
                        "seed": seed,
                        "hyperparameter": "reward_scaling",
                        "hyperparameter_value": reward_scaling,
                    }
                )
    for lr in ("3e-4", "5e-4", "1e-3"):
        for exploration_noise in ("0.05", "0.1", "0.2"):
            for seed in seeds:
                rows.append(
                    {
                        "algo": "td3",
                        "lr": lr,
                        "seed": seed,
                        "hyperparameter": "exploration_noise",
                        "hyperparameter_value": exploration_noise,
                    }
                )
    for lr in ("1e-4", "1e-3", "3e-3"):
        for ent_coef in ("0.0", "0.01"):
            for seed in seeds:
                rows.append(
                    {
                        "algo": "ppo",
                        "lr": lr,
                        "seed": seed,
                        "hyperparameter": "ent_coef",
                        "hyperparameter_value": ent_coef,
                    }
                )
    return rows


def _cell_id(cell: dict[str, str | int]) -> str:
    return (
        f"{cell['algo']}_lr{cell['lr']}_hp-{cell['hyperparameter']}"
        f"_val-{_label_value(str(cell['hyperparameter_value']))}_seed{cell['seed']}"
    )


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="results/final_sweep")
    parser.add_argument("--out", default="results/final_sweep/summary.csv")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    rows: list[dict[str, object]] = []
    missing_cells: list[str] = []
    missing_splits: list[tuple[str, str]] = []

    if not sweep_root.exists():
        raise SystemExit(f"sweep root does not exist: {sweep_root}")

    for cell in _expected_cells():
        cell_id = _cell_id(cell)
        cell_dir = sweep_root / cell_id
        meta = _read_json(cell_dir / "meta.json")
        if meta is None:
            missing_cells.append(cell_id)
            continue

        train = _read_json(cell_dir / "train.json")
        if train is None:
            missing_splits.append((cell_id, "public_dev"))
        else:
            rows.append(
                {
                    **meta,
                    "split": "public_dev",
                    "run_id": train.get("run_id"),
                    "average_score": train.get("average_score"),
                }
            )

        for split in EVAL_SPLITS:
            payload = _read_json(cell_dir / f"eval_{split}.json")
            if payload is None:
                missing_splits.append((cell_id, split))
                continue
            rows.append(
                {
                    **meta,
                    "split": split,
                    "run_id": payload.get("run_id"),
                    "average_score": payload.get("average_score"),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_id",
        "algo",
        "lr",
        "seed",
        "hyperparameter",
        "hyperparameter_value",
        "split",
        "run_id",
        "average_score",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out_path}")
    if missing_cells:
        print(f"MISSING CELLS ({len(missing_cells)}):")
        for cell_id in missing_cells:
            print(f"  {cell_id}")
    if missing_splits:
        print(f"MISSING SPLITS ({len(missing_splits)}):")
        for cell_id, split in missing_splits:
            print(f"  {cell_id} split={split}")
    if (missing_cells or missing_splits) and not args.allow_missing:
        raise SystemExit("final sweep is incomplete")


if __name__ == "__main__":
    main()

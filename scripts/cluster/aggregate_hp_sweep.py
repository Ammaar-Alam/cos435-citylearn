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
DEFAULT_LRS = ("1e-4", "3e-4", "1e-3")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_HYPERPARAMETER = "residual_scaling"
DEFAULT_HYPERPARAMETER_VALUES = ("0.5", "0.75", "1.0")


def _label_value(value: str) -> str:
    return value.replace("-", "m").replace(".", "p")


def _parse_ints(value: str) -> list[int]:
    if "-" in value:
        lo, hi = (int(part) for part in value.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(part) for part in value.split(",") if part]


def _expected_cells(
    *,
    algo: str,
    lrs: list[str],
    seeds: list[int],
    hyperparameter: str,
    hyperparameter_values: list[str],
) -> list[str]:
    return [
        (
            f"{algo}_lr{lr}_hp-{hyperparameter}"
            f"_val-{_label_value(str(hyperparameter_value))}_seed{seed}"
        )
        for lr in lrs
        for hyperparameter_value in hyperparameter_values
        for seed in seeds
    ]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _rows_for_cell(
    cell_dir: Path,
    eval_splits: list[str],
) -> tuple[list[dict[str, object]], list[str], bool]:
    meta = _read_json(cell_dir / "meta.json")
    if meta is None:
        return [], [], True

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    train = _read_json(cell_dir / "train.json")
    if train is None:
        missing.append("public_dev")
    else:
        rows.append(
            {
                **meta,
                "split": "public_dev",
                "run_id": train.get("run_id"),
                "average_score": train.get("average_score"),
            }
        )

    for split in eval_splits:
        payload = _read_json(cell_dir / f"eval_{split}.json")
        if payload is None:
            missing.append(split)
            continue
        rows.append(
            {
                **meta,
                "split": split,
                "run_id": payload.get("run_id"),
                "average_score": payload.get("average_score"),
            }
        )

    return rows, missing, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="results/residual_sac_sweep")
    parser.add_argument("--out", default="results/residual_sac_sweep/summary.csv")
    parser.add_argument(
        "--eval-splits",
        default=",".join(EVAL_SPLITS),
        help="comma-separated held-out splits expected under each cell directory",
    )
    parser.add_argument(
        "--algo",
        default="sac_residual",
        help="algorithm label expected in residual sweep cell ids",
    )
    parser.add_argument(
        "--lrs",
        default=",".join(DEFAULT_LRS),
        help="comma-separated learning rates expected in the sweep",
    )
    parser.add_argument(
        "--seeds",
        default=f"{DEFAULT_SEEDS[0]}-{DEFAULT_SEEDS[-1]}",
        help="seed range, e.g. 0-2, or comma-separated seed list",
    )
    parser.add_argument(
        "--hyperparameter",
        default=DEFAULT_HYPERPARAMETER,
        help="hyperparameter name encoded in cell ids",
    )
    parser.add_argument(
        "--hyperparameter-values",
        default=",".join(DEFAULT_HYPERPARAMETER_VALUES),
        help="comma-separated hyperparameter values expected in the sweep",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        raise SystemExit(f"sweep root does not exist: {sweep_root}")

    eval_splits = [split for split in args.eval_splits.split(",") if split]
    rows: list[dict[str, object]] = []
    missing_cells: list[str] = []
    missing_metadata: list[str] = []
    missing: list[tuple[str, str]] = []
    expected_cells = _expected_cells(
        algo=args.algo,
        lrs=[lr for lr in args.lrs.split(",") if lr],
        seeds=_parse_ints(args.seeds),
        hyperparameter=args.hyperparameter,
        hyperparameter_values=[
            value for value in args.hyperparameter_values.split(",") if value
        ],
    )

    for cell_id in expected_cells:
        cell_dir = sweep_root / cell_id
        if not cell_dir.exists():
            missing_cells.append(cell_id)
            continue
        cell_rows, cell_missing, cell_missing_metadata = _rows_for_cell(cell_dir, eval_splits)
        if cell_missing_metadata:
            missing_metadata.append(cell_id)
            continue
        rows.extend(cell_rows)
        missing.extend((cell_id, split) for split in cell_missing)

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
        "expert_policy",
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
    if missing_metadata:
        print(f"MISSING METADATA ({len(missing_metadata)}):")
        for cell_id in missing_metadata:
            print(f"  {cell_id}")
    if missing:
        print(f"MISSING SPLITS ({len(missing)}):")
        for cell_id, split in missing:
            print(f"  {cell_id} split={split}")
    if (missing_cells or missing_metadata or missing) and not args.allow_missing:
        raise SystemExit("hyperparameter sweep is incomplete")


if __name__ == "__main__":
    main()

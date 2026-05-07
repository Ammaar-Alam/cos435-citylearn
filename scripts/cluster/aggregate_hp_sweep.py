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


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _rows_for_cell(
    cell_dir: Path,
    eval_splits: list[str],
) -> tuple[list[dict[str, object]], list[str]]:
    meta = _read_json(cell_dir / "meta.json")
    if meta is None:
        return [], []

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

    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="results/residual_sac_sweep")
    parser.add_argument("--out", default="results/residual_sac_sweep/summary.csv")
    parser.add_argument(
        "--eval-splits",
        default=",".join(EVAL_SPLITS),
        help="comma-separated held-out splits expected under each cell directory",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        raise SystemExit(f"sweep root does not exist: {sweep_root}")

    eval_splits = [split for split in args.eval_splits.split(",") if split]
    rows: list[dict[str, object]] = []
    missing: list[tuple[str, str]] = []

    for cell_dir in sorted(path for path in sweep_root.iterdir() if path.is_dir()):
        cell_rows, cell_missing = _rows_for_cell(cell_dir, eval_splits)
        rows.extend(cell_rows)
        missing.extend((cell_dir.name, split) for split in cell_missing)

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
    if missing:
        print(f"MISSING SPLITS ({len(missing)}):")
        for cell_id, split in missing:
            print(f"  {cell_id} split={split}")
    if missing and not args.allow_missing:
        raise SystemExit("hyperparameter sweep is incomplete")


if __name__ == "__main__":
    main()

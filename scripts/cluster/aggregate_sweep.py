from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# matches cell dirs like "ppo_lr3e-4_seed0" or "sac_lr1e-4_seed9"
CELL_PATTERN = re.compile(r"^(?P<algo>[a-z]+)_lr(?P<lr>[^_]+)_seed(?P<seed>\d+)$")


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root",
        default="results/sweep",
        help="directory containing <algo>_lr<lr>_seed<n>/ subdirs",
    )
    parser.add_argument("--out", default="results/sweep/summary.csv")
    parser.add_argument(
        "--algos",
        default="ppo,sac",
        help="comma-separated algos expected in the sweep",
    )
    parser.add_argument("--lrs", default="1e-4,3e-4", help="comma-separated lrs expected per algo")
    parser.add_argument("--seeds", default="0-9", help="seed range (e.g. 0-9) or comma list")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit 0 even when cells are missing; by default we fail loud",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    rows: list[dict] = []
    found_cells: set[tuple[str, str, int]] = set()

    if not sweep_root.exists():
        raise SystemExit(f"sweep root does not exist: {sweep_root}")

    for cell_dir in sorted(sweep_root.iterdir()):
        match = CELL_PATTERN.match(cell_dir.name)
        if match is None:
            continue
        algo = match["algo"]
        lr = match["lr"]
        seed = int(match["seed"])
        found_cells.add((algo, lr, seed))

        train = _read(cell_dir / "train.json")
        if train is None:
            continue
        rows.append({
            "algo": algo,
            "lr": lr,
            "seed": seed,
            "split": "public_dev",
            "run_id": train.get("run_id"),
            "average_score": train.get("average_score"),
        })
        for split in ("phase_3_1", "phase_3_2", "phase_3_3"):
            eval_payload = _read(cell_dir / f"eval_{split}.json")
            if eval_payload is None:
                continue
            rows.append({
                "algo": algo,
                "lr": lr,
                "seed": seed,
                "split": split,
                "run_id": eval_payload.get("run_id"),
                "average_score": eval_payload.get("average_score"),
            })

    expected_algos = [a for a in args.algos.split(",") if a]
    expected_lrs = [lr for lr in args.lrs.split(",") if lr]
    if "-" in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-", 1))
        expected_seeds = list(range(lo, hi + 1))
    else:
        expected_seeds = [int(s) for s in args.seeds.split(",") if s]

    expected_cells = {
        (algo, lr, seed)
        for algo in expected_algos
        for lr in expected_lrs
        for seed in expected_seeds
    }
    missing_cells = sorted(expected_cells - found_cells)
    missing_splits: list[tuple[str, str, int, str]] = []
    for (algo, lr, seed) in sorted(found_cells):
        cell_dir = sweep_root / f"{algo}_lr{lr}_seed{seed}"
        if _read(cell_dir / "train.json") is None:
            missing_splits.append((algo, lr, seed, "public_dev (train.json)"))
        for split in ("phase_3_1", "phase_3_2", "phase_3_3"):
            if _read(cell_dir / f"eval_{split}.json") is None:
                missing_splits.append((algo, lr, seed, split))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["algo", "lr", "seed", "split", "run_id", "average_score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out_path}")

    if missing_cells:
        print(f"MISSING CELLS ({len(missing_cells)}):")
        for algo, lr, seed in missing_cells:
            print(f"  {algo} lr={lr} seed={seed}")
    if missing_splits:
        print(f"MISSING SPLITS ({len(missing_splits)}):")
        for algo, lr, seed, split in missing_splits:
            print(f"  {algo} lr={lr} seed={seed} split={split}")

    if (missing_cells or missing_splits) and not args.allow_missing:
        raise SystemExit(
            "sweep is incomplete; rerun the missing cells/splits or pass --allow-missing"
        )


if __name__ == "__main__":
    main()

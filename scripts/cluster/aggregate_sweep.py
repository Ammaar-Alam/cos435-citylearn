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
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    rows: list[dict] = []
    for cell_dir in sorted(sweep_root.iterdir()):
        match = CELL_PATTERN.match(cell_dir.name)
        if match is None:
            continue
        algo = match["algo"]
        lr = match["lr"]
        seed = int(match["seed"])

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


if __name__ == "__main__":
    main()

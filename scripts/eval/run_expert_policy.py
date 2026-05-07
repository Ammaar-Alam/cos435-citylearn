from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from cos435_citylearn.algorithms.sac import resolve_reward_function
from cos435_citylearn.algorithms.sac.expert import (
    SUPPORTED_EXPERT_POLICIES,
    ExpertActionPolicy,
)
from cos435_citylearn.config import resolve_path
from cos435_citylearn.env import PerBuildingEnvAdapter, make_citylearn_env
from cos435_citylearn.eval import build_metrics_payload, flatten_metrics_row
from cos435_citylearn.io import ensure_parent, write_json


def _evaluate_split(
    *,
    policy_name: str,
    split: str,
    seed: int,
    reward_version: str,
    max_steps: int | None,
) -> dict[str, Any]:
    reward_function = resolve_reward_function(reward_version)
    env_bundle = make_citylearn_env(
        "configs/env/citylearn_2023.yaml",
        f"configs/splits/{split}.yaml",
        seed=seed,
        central_agent=False,
        reward_function=reward_function,
    )
    adapter = PerBuildingEnvAdapter(env_bundle.env)
    observations = adapter.reset()
    policy = ExpertActionPolicy(
        policy_name,
        observation_names=env_bundle.env.observation_names,
        action_names=env_bundle.env.action_names,
        action_lows=[space.low.astype(float).tolist() for space in env_bundle.env.action_space],
        action_highs=[space.high.astype(float).tolist() for space in env_bundle.env.action_space],
    )
    step_count = 0

    while not adapter.done:
        if max_steps is not None and step_count >= max_steps:
            break
        actions = adapter.clip_actions(policy.predict(observations))
        result = adapter.step(actions)
        observations = result.observations
        step_count += 1
        if result.terminated:
            break

    run_id = f"expert__{policy_name}__{split}__seed{seed}"
    metrics = build_metrics_payload(
        env_bundle.env,
        {
            "algorithm": "expert",
            "variant": policy_name,
            "split": split,
            "seed": env_bundle.seed,
            "dataset_name": env_bundle.dataset_name,
            "run_id": run_id,
        },
    )
    metrics["step_count"] = step_count
    if max_steps is not None:
        metrics["max_steps"] = max_steps
    return metrics


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    target = ensure_parent(path)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="adaptive_storage_v1",
        choices=sorted(SUPPORTED_EXPERT_POLICIES),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["public_dev"],
        help="split stems under configs/splits",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-version", default="reward_v2")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="debug-only partial rollout limit; omit for complete official-style eval",
    )
    parser.add_argument(
        "--summary-out",
        default="results/metrics/expert_policy_summary.csv",
        help="CSV summary output path",
    )
    parser.add_argument(
        "--payload-dir",
        default=None,
        help="optional directory for one metrics JSON payload per split",
    )
    args = parser.parse_args()

    payloads = [
        _evaluate_split(
            policy_name=args.policy,
            split=split,
            seed=args.seed,
            reward_version=args.reward_version,
            max_steps=args.max_steps,
        )
        for split in args.splits
    ]
    rows = [
        {
            **flatten_metrics_row(payload),
            "step_count": payload["step_count"],
            "max_steps": payload.get("max_steps"),
        }
        for payload in payloads
    ]

    summary_path = resolve_path(args.summary_out)
    _write_summary(summary_path, rows)
    if args.payload_dir is not None:
        payload_dir = resolve_path(args.payload_dir)
        for payload in payloads:
            write_json(payload_dir / f"{payload['run_id']}.json", payload)

    print(json.dumps({"summary_path": str(summary_path), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

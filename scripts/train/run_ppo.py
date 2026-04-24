from __future__ import annotations

import argparse
import json

from cos435_citylearn.baselines.ppo import run_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train/ppo/ppo_central_baseline.yaml",
        help="training config path",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/eval/default.yaml",
        help="evaluation config path",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "override env.split (eg. phase_3_1). required when using "
            "--artifact-id to eval on a held-out split"
        ),
    )
    parser.add_argument(
        "--artifact-id",
        default=None,
        help="run_id of an existing PPO run to re-evaluate without training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override training.seed from the config (useful for multi-seed sweeps)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override training.learning_rate from the config",
    )
    parser.add_argument(
        "--allow-cross-reward-eval",
        action="store_true",
        help=(
            "opt into evaluating a checkpoint whose trained variant/reward/features "
            "version does not match the eval config (mismatch is recorded in the "
            "run manifest under runtime_label_mismatches)"
        ),
    )
    args = parser.parse_args()

    payload = run_ppo(
        config_path=args.config,
        eval_config_path=args.eval_config,
        artifact_id=args.artifact_id,
        split_override=args.split,
        seed_override=args.seed,
        lr_override=args.lr,
        allow_cross_reward_eval=args.allow_cross_reward_eval,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

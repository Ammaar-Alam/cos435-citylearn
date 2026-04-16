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
    args = parser.parse_args()
    payload = run_ppo(config_path=args.config, eval_config_path=args.eval_config)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

import argparse
import json

from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.smoke import run_random_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env/citylearn_2023.yaml")
    parser.add_argument("--split-config", default="configs/splits/public_dev.yaml")
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run_random_rollout(
        env_config_path=args.env_config,
        split_config_path=args.split_config,
        max_steps=args.max_steps,
        seed=args.seed,
        trace_output_path=RESULTS_DIR / "manifests" / "random_rollout_trace.json",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

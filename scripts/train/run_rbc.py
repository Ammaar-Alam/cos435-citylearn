import argparse
import json

from cos435_citylearn.baselines import run_rbc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/rbc/rbc_builtin.yaml")
    parser.add_argument("--eval-config", default="configs/eval/default.yaml")
    args = parser.parse_args()
    payload = run_rbc(config_path=args.config, eval_config_path=args.eval_config)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

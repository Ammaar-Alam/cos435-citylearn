import argparse
import json

from cos435_citylearn.env import write_env_schema_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env/citylearn_2023.yaml")
    parser.add_argument("--split-config", default="configs/splits/public_dev.yaml")
    args = parser.parse_args()
    payload = write_env_schema_manifest(
        env_config_path=args.env_config,
        split_config_path=args.split_config,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

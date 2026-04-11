import sys
from pathlib import Path

import yaml

from cos435_citylearn.paths import CONFIGS_DIR

TRAIN_REQUIRED = {"env", "algorithm", "reward", "features", "training", "evaluation", "logging"}
EVAL_REQUIRED = {"env", "evaluation", "logging"}
ENV_REQUIRED = {"env"}
SPLIT_REQUIRED = {"split"}


def required_keys(path: Path) -> set[str]:
    path_str = path.as_posix()
    if "/train/" in path_str:
        return TRAIN_REQUIRED
    if "/eval/" in path_str:
        return EVAL_REQUIRED
    if "/env/" in path_str:
        return ENV_REQUIRED
    if "/splits/" in path_str:
        return SPLIT_REQUIRED
    return set()


def main() -> None:
    failures: list[str] = []
    checked = 0

    for path in sorted(CONFIGS_DIR.rglob("*.yaml")):
        checked += 1
        data = yaml.safe_load(path.read_text()) or {}
        expected = required_keys(path)
        missing = sorted(expected - set(data))
        if missing:
            rel_path = path.relative_to(CONFIGS_DIR.parent)
            missing_keys = ", ".join(missing)
            failures.append(f"{rel_path} missing top-level keys: {missing_keys}")

    if failures:
        print("config check failed", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"checked {checked} config files")


if __name__ == "__main__":
    main()

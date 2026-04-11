import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from cos435_citylearn.paths import CONFIGS_DIR, DATA_DIR, REPO_ROOT, RESULTS_DIR


def maybe_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": str(REPO_ROOT),
        "configs_dir": str(CONFIGS_DIR),
        "data_dir": str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "packages": {
            "CityLearn": maybe_version("CityLearn"),
            "stable-baselines3": maybe_version("stable-baselines3"),
            "torch": maybe_version("torch"),
            "pytest": maybe_version("pytest"),
            "ruff": maybe_version("ruff"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

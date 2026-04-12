import json
from importlib.metadata import PackageNotFoundError, version

from cos435_citylearn.paths import CONFIGS_DIR, DATA_DIR, REPO_ROOT, RESULTS_DIR
from cos435_citylearn.runtime import build_environment_lock


def maybe_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    dataset_manifest = DATA_DIR / "manifests" / "citylearn_2023_manifest.json"
    default_schema = (
        DATA_DIR
        / "external"
        / "citylearn_2023"
        / "citylearn_challenge_2023_phase_2_local_evaluation"
        / "schema.json"
    )
    payload = build_environment_lock(
        {
            "repo_root": str(REPO_ROOT),
            "configs_dir": str(CONFIGS_DIR),
            "data_dir": str(DATA_DIR),
            "results_dir": str(RESULTS_DIR),
            "default_schema_present": default_schema.exists(),
            "dataset_manifest_present": dataset_manifest.exists(),
            "packages": {
                "CityLearn": maybe_version("CityLearn"),
                "stable-baselines3": maybe_version("stable-baselines3"),
                "torch": maybe_version("torch"),
                "pytest": maybe_version("pytest"),
                "ruff": maybe_version("ruff"),
            },
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

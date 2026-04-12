from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citylearn.citylearn import CityLearnEnv

from cos435_citylearn.config import load_yaml, resolve_path
from cos435_citylearn.dataset import DEFAULT_DATASET_NAME
from cos435_citylearn.io import write_json
from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.runtime import build_environment_lock


@dataclass
class EnvBundle:
    env: CityLearnEnv
    env_config_path: Path
    split_config_path: Path | None
    dataset_name: str
    schema_path: Path
    seed: int
    central_agent: bool


def _load_configs(
    env_config_path: str | Path,
    split_config_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], Path | None, dict[str, Any]]:
    resolved_env_path = resolve_path(env_config_path)
    env_config = load_yaml(resolved_env_path)

    if split_config_path is None:
        return resolved_env_path, env_config, None, {}

    resolved_split_path = resolve_path(split_config_path)
    split_config = load_yaml(resolved_split_path)
    return resolved_env_path, env_config, resolved_split_path, split_config


def resolve_schema_path(
    env_config_path: str | Path,
    split_config_path: str | Path | None = None,
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    resolved_env_path, env_config, resolved_split_path, split_config = _load_configs(
        env_config_path,
        split_config_path,
    )
    env_settings = env_config["env"]
    split_settings = split_config.get("split", {})
    dataset_name = split_settings.get(
        "dataset_name",
        env_settings.get("default_dataset", DEFAULT_DATASET_NAME),
    )
    dataset_root = resolve_path(env_settings["dataset_root"])
    schema_path = dataset_root / dataset_name / "schema.json"
    return schema_path, dataset_name, env_config, split_config


def make_citylearn_env(
    env_config_path: str | Path = "configs/env/citylearn_2023.yaml",
    split_config_path: str | Path | None = "configs/splits/public_dev.yaml",
    seed: int | None = None,
    central_agent: bool | None = None,
) -> EnvBundle:
    resolved_env_path, env_config, resolved_split_path, split_config = _load_configs(
        env_config_path,
        split_config_path,
    )
    env_settings = env_config["env"]
    split_settings = split_config.get("split", {})
    dataset_name = split_settings.get(
        "dataset_name",
        env_settings.get("default_dataset", DEFAULT_DATASET_NAME),
    )
    dataset_root = resolve_path(env_settings["dataset_root"])
    schema_path = dataset_root / dataset_name / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"CityLearn schema not found at {schema_path}. "
            "run `make download-citylearn` first"
        )

    resolved_seed = int(env_settings["seed"] if seed is None else seed)
    resolved_central_agent = bool(
        split_settings.get("central_agent", env_settings["central_agent"])
        if central_agent is None
        else central_agent
    )
    buildings = split_settings.get("buildings") or None
    shared_observations = env_settings.get("shared_observations") or None
    episode_time_steps = env_settings.get("episode_time_steps")
    rolling_episode_split = env_settings.get("rolling_episode_split")
    random_episode_split = env_settings.get("random_episode_split")

    env = CityLearnEnv(
        str(schema_path),
        central_agent=resolved_central_agent,
        random_seed=resolved_seed,
        buildings=buildings,
        shared_observations=shared_observations,
        episode_time_steps=episode_time_steps,
        rolling_episode_split=rolling_episode_split,
        random_episode_split=random_episode_split,
    )

    return EnvBundle(
        env=env,
        env_config_path=resolved_env_path,
        split_config_path=resolved_split_path,
        dataset_name=dataset_name,
        schema_path=schema_path,
        seed=resolved_seed,
        central_agent=resolved_central_agent,
    )


def get_env_metadata(bundle: EnvBundle) -> dict[str, Any]:
    env = bundle.env
    reset_sample = env.reset()
    observation_sample = reset_sample[0] if isinstance(reset_sample, tuple) else reset_sample
    action_space = []
    observation_space = []

    for box in env.action_space:
        action_space.append(
            {
                "shape": list(box.shape),
                "low": box.low.astype(float).tolist(),
                "high": box.high.astype(float).tolist(),
            }
        )

    for box in env.observation_space:
        observation_space.append(
            {
                "shape": list(box.shape),
                "low": box.low.astype(float).tolist(),
                "high": box.high.astype(float).tolist(),
            }
        )

    return {
        "dataset_name": bundle.dataset_name,
        "schema_path": str(bundle.schema_path),
        "central_agent": bundle.central_agent,
        "seed": bundle.seed,
        "time_steps": int(env.time_steps),
        "building_names": [building.name for building in env.buildings],
        "observation_names": env.observation_names,
        "action_names": env.action_names,
        "observation_dimensions": [len(names) for names in env.observation_names],
        "action_dimensions": [len(names) for names in env.action_names],
        "observation_space": observation_space,
        "action_space": action_space,
        "observation_sample_ranges": [
            {
                "min": min(float(value) for value in values),
                "max": max(float(value) for value in values),
            }
            for values in observation_sample
        ],
    }


def write_env_schema_manifest(
    env_config_path: str | Path = "configs/env/citylearn_2023.yaml",
    split_config_path: str | Path = "configs/splits/public_dev.yaml",
    schema_output_path: str | Path | None = None,
    environment_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle = make_citylearn_env(env_config_path, split_config_path)
    schema_output = (
        RESULTS_DIR / "manifests" / "observation_action_schema.json"
        if schema_output_path is None
        else Path(schema_output_path)
    )
    environment_lock_output = (
        RESULTS_DIR / "manifests" / "environment_lock.json"
        if environment_lock_path is None
        else Path(environment_lock_path)
    )
    metadata = get_env_metadata(bundle)
    write_json(schema_output, metadata)
    write_json(
        environment_lock_output,
        build_environment_lock(
            {
                "dataset_name": bundle.dataset_name,
                "schema_path": str(bundle.schema_path),
                "seed": bundle.seed,
                "central_agent": bundle.central_agent,
            }
        ),
    )
    return metadata

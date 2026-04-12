from __future__ import annotations

from pathlib import Path
from typing import Any

from cos435_citylearn.env.adapters import CentralizedEnvAdapter, PerBuildingEnvAdapter
from cos435_citylearn.env.loader import make_citylearn_env
from cos435_citylearn.io import write_json


def _adapter_for_bundle(bundle):
    if bundle.central_agent:
        return CentralizedEnvAdapter(bundle.env)

    return PerBuildingEnvAdapter(bundle.env)


def run_random_rollout(
    env_config_path: str | Path = "configs/env/citylearn_2023.yaml",
    split_config_path: str | Path = "configs/splits/public_dev.yaml",
    max_steps: int = 48,
    seed: int | None = None,
    trace_output_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle = make_citylearn_env(env_config_path, split_config_path, seed=seed)
    adapter = _adapter_for_bundle(bundle)
    adapter.reset()
    trace = []
    no_nan_observations = True
    no_nan_rewards = True
    steps = 0

    while not adapter.done and steps < max_steps:
        actions = adapter.sample_action(seed=bundle.seed + steps)
        clipped_actions = adapter.clip_actions(actions)
        result = adapter.step(clipped_actions)
        no_nan_observations = no_nan_observations and all(
            value == value for row in result.observations for value in row
        )
        no_nan_rewards = no_nan_rewards and all(value == value for value in result.rewards)
        trace.append(
            {
                "step": steps,
                "actions": clipped_actions,
                "rewards": result.rewards,
                "terminated": result.terminated,
            }
        )
        steps += 1

    payload = {
        "dataset_name": bundle.dataset_name,
        "seed": bundle.seed,
        "steps": steps,
        "terminated": adapter.done,
        "no_nan_observations": no_nan_observations,
        "no_nan_rewards": no_nan_rewards,
        "trace": trace,
    }

    if trace_output_path is not None:
        write_json(trace_output_path, payload)

    return payload

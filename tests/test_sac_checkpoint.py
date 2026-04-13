from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cos435_citylearn.baselines.sac import (
    _build_adapter,
    _build_checkpoint_payload,
    _instantiate_controller,
    _instantiate_controller_from_checkpoint,
    _maybe_reset_reward,
)
from cos435_citylearn.config import load_yaml
from cos435_citylearn.env import make_citylearn_env
from cos435_citylearn.algorithms.sac import resolve_reward_function
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


def _warm_checkpoint_payload(config_path: str, tmp_path: Path):
    config = load_yaml(config_path)
    reward_function = resolve_reward_function(config["reward"]["version"])
    central_agent = config["algorithm"]["control_mode"] == "centralized"
    env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=central_agent,
        reward_function=reward_function,
    )
    adapter = _build_adapter(env_bundle.env)
    controller = _instantiate_controller(env_bundle.env, config)
    observations = adapter.reset()
    _maybe_reset_reward(env_bundle.env)

    for _ in range(20):
        actions = controller.predict(observations, deterministic=False)
        clipped_actions = adapter.clip_actions(actions)
        result = adapter.step(clipped_actions)
        controller.update(
            observations,
            clipped_actions,
            result.rewards,
            result.observations,
            done=result.terminated,
        )
        observations = result.observations
        if result.terminated:
            observations = adapter.reset()
            _maybe_reset_reward(env_bundle.env)

    reference_actions = controller.predict(observations, deterministic=True)
    checkpoint_payload = _build_checkpoint_payload(
        run_id="checkpoint_test",
        config=config,
        env_bundle=env_bundle,
        controller=controller,
        training_step=20,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    return config, observations, reference_actions, checkpoint_path


def _assert_checkpoint_roundtrip(config_path: str, tmp_path: Path) -> None:
    config, observations, reference_actions, checkpoint_path = _warm_checkpoint_payload(
        config_path, tmp_path
    )
    loaded_payload = torch.load(checkpoint_path, map_location="cpu")
    reward_function = resolve_reward_function(config["reward"]["version"])
    central_agent = config["algorithm"]["control_mode"] == "centralized"
    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=central_agent,
        reward_function=reward_function,
    )
    reloaded_controller = _instantiate_controller_from_checkpoint(
        eval_env_bundle.env,
        loaded_payload,
    )
    restored_actions = reloaded_controller.predict(observations, deterministic=True)

    assert len(restored_actions) == len(reference_actions)
    for expected, actual in zip(reference_actions, restored_actions):
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_centralized_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    _assert_checkpoint_roundtrip("configs/train/sac/sac_central_smoke.yaml", tmp_path)


def test_shared_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    _assert_checkpoint_roundtrip("configs/train/sac/sac_shared_dtde_smoke.yaml", tmp_path)

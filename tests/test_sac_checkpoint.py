from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cos435_citylearn.algorithms.sac.checkpoints import (
    safe_load_checkpoint_payload,
    validate_checkpoint_payload_structure,
    validate_checkpoint_env_compatibility,
    validate_checkpoint_runner_compatibility,
)
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


def _minimal_central_checkpoint_payload() -> dict[str, object]:
    return {
        "algorithm": "sac",
        "control_mode": "centralized",
        "observation_names": [["hour", "load"]],
        "action_names": [["battery"]],
        "controller_state": {
            "controller_type": "centralized_native",
            "hidden_dimension": [64, 64],
            "discount": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "lr": 0.0003,
            "batch_size": 16,
            "replay_buffer_capacity": 128,
            "standardize_start_time_step": 8,
            "end_exploration_time_step": 8,
            "action_scaling_coefficient": 0.5,
            "reward_scaling": 5.0,
            "update_per_time_step": 1,
            "normalized": [False],
            "policy_state_dicts": [{}],
            "soft_q1_state_dicts": [{}],
            "soft_q2_state_dicts": [{}],
            "target_soft_q1_state_dicts": [{}],
            "target_soft_q2_state_dicts": [{}],
            "policy_optimizer_state_dicts": [{}],
            "soft_q_optimizer1_state_dicts": [{}],
            "soft_q_optimizer2_state_dicts": [{}],
            "norm_mean": [None],
            "norm_std": [None],
            "r_norm_mean": [None],
            "r_norm_std": [None],
        },
    }


def _minimal_shared_checkpoint_payload() -> dict[str, object]:
    return {
        "algorithm": "sac",
        "control_mode": "shared_dtde",
        "observation_names": [["hour", "load"]],
        "action_names": [["battery"]],
        "controller_state": {
            "controller_type": "shared_parameter_sac",
            "hidden_dimension": [64, 64],
            "discount": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "lr": 0.0003,
            "batch_size": 16,
            "replay_buffer_capacity": 128,
            "standardize_start_time_step": 8,
            "end_exploration_time_step": 8,
            "action_scaling_coefficient": 0.5,
            "reward_scaling": 5.0,
            "update_per_time_step": 1,
            "normalized": False,
            "policy_state_dict": {},
            "soft_q1_state_dict": {},
            "soft_q2_state_dict": {},
            "target_soft_q1_state_dict": {},
            "target_soft_q2_state_dict": {},
            "policy_optimizer_state_dict": {},
            "soft_q_optimizer1_state_dict": {},
            "soft_q_optimizer2_state_dict": {},
            "norm_mean": None,
            "norm_std": None,
            "r_norm_mean": None,
            "r_norm_std": None,
            "shared_context_dimension": 4,
        },
    }


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


def test_safe_load_checkpoint_payload_uses_weights_only(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")
    payload = _minimal_central_checkpoint_payload()

    def fake_load(path, *, map_location, weights_only):
        assert Path(path) == checkpoint_path
        assert map_location == "cpu"
        assert weights_only is True
        return payload

    monkeypatch.setattr(
        "cos435_citylearn.algorithms.sac.checkpoints.torch.load",
        fake_load,
    )

    loaded = safe_load_checkpoint_payload(checkpoint_path)
    assert loaded == payload


def test_validate_checkpoint_runner_compatibility_rejects_control_mode_mismatch() -> None:
    payload = _minimal_central_checkpoint_payload()
    payload["control_mode"] = "shared_dtde"
    config = {
        "algorithm": {
            "name": "sac",
            "control_mode": "centralized",
        }
    }

    with pytest.raises(ValueError, match="control_mode"):
        validate_checkpoint_runner_compatibility(payload, config)


def test_validate_checkpoint_env_compatibility_rejects_schema_mismatch() -> None:
    payload = _minimal_central_checkpoint_payload()

    with pytest.raises(ValueError, match="observation schema"):
        validate_checkpoint_env_compatibility(
            payload,
            observation_names=[["hour", "different_load"]],
            action_names=[["battery"]],
        )


def test_validate_checkpoint_payload_structure_rejects_missing_shared_context_dimension() -> None:
    payload = _minimal_shared_checkpoint_payload()
    del payload["controller_state"]["shared_context_dimension"]

    with pytest.raises(ValueError, match="shared_context_dimension"):
        validate_checkpoint_payload_structure(payload)


def test_shared_controller_rejects_non_default_shared_context_width() -> None:
    require_benchmark_runtime()
    require_dataset()
    config = load_yaml("configs/train/sac/sac_shared_dtde_smoke.yaml")
    config["features"]["shared_context_dimension"] = 5
    reward_function = resolve_reward_function(config["reward"]["version"])
    env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=False,
        reward_function=reward_function,
    )

    with pytest.raises(ValueError, match="shared_context_dimension"):
        _instantiate_controller(env_bundle.env, config)


def test_centralized_checkpoint_roundtrip_rejects_truncated_state_lists(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    config, _, _, checkpoint_path = _warm_checkpoint_payload(
        "configs/train/sac/sac_central_smoke.yaml", tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    controller_state = payload["controller_state"]
    controller_state["policy_state_dicts"] = controller_state["policy_state_dicts"][:-1]

    reward_function = resolve_reward_function(config["reward"]["version"])
    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=True,
        reward_function=reward_function,
    )

    with pytest.raises(ValueError, match="policy_state_dicts count"):
        _instantiate_controller_from_checkpoint(eval_env_bundle.env, payload)


def test_centralized_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    _assert_checkpoint_roundtrip("configs/train/sac/sac_central_smoke.yaml", tmp_path)


def test_shared_checkpoint_roundtrip(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    _assert_checkpoint_roundtrip("configs/train/sac/sac_shared_dtde_smoke.yaml", tmp_path)

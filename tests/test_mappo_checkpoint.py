from __future__ import annotations

import pytest

from cos435_citylearn.algorithms.mappo.checkpoints import (
    validate_mappo_checkpoint_env_compatibility,
    validate_mappo_checkpoint_payload_structure,
    validate_mappo_checkpoint_runner_compatibility,
)


def _minimal_payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "algorithm": "mappo",
        "control_mode": "shared_ctde",
        "variant": "mappo_shared_ctde_reward_v2",
        "reward_version": "reward_v2",
        "features_version": "centralized_critic_context_v1",
        "observation_names": [["hour", "load"]] * 3,
        "action_names": [["battery"]] * 3,
        "controller_state": {
            "controller_type": "forecast_augmented_mappo",
            "hidden_dimension": [32, 32],
            "critic_hidden_dimension": [64, 64],
            "lr": 3e-4,
            "clip_range": 0.2,
            "n_epochs": 2,
            "minibatch_size": 16,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "rollout_steps": 32,
            "reward_scaling": 1.0,
            "shared_context_dimension": 4,
            "shared_context_version": "v2",
            "critic_context_dimension": 12,
            "critic_context_version": "centralized_critic_context_v1",
            "normalize_observations": True,
            "normalize_critic_observations": True,
            "normalize_rewards": True,
            "normalize_advantage": True,
            "target_kl": None,
            "policy_state_dict": {},
            "value_state_dict": {},
            "policy_optimizer_state_dict": {},
            "value_optimizer_state_dict": {},
            "obs_rms_mean": None,
            "obs_rms_var": None,
            "obs_rms_count": 0.0,
            "critic_obs_rms_mean": None,
            "critic_obs_rms_var": None,
            "critic_obs_rms_count": 0.0,
            "reward_rms_mean": 0.0,
            "reward_rms_var": None,
            "reward_rms_count": 0.0,
            "time_step": 0,
            "total_updates": 0,
        },
    }


def test_structure_validator_accepts_valid_mappo_payload() -> None:
    validate_mappo_checkpoint_payload_structure(_minimal_payload())


def test_structure_validator_rejects_wrong_algorithm() -> None:
    payload = _minimal_payload()
    payload["algorithm"] = "ppo"
    with pytest.raises(ValueError, match="not a MAPPO checkpoint"):
        validate_mappo_checkpoint_payload_structure(payload)


def test_runner_compatibility_passes_on_matching_runtime_labels() -> None:
    payload = _minimal_payload()
    config = {
        "algorithm": {
            "name": "mappo",
            "control_mode": "shared_ctde",
            "variant": "mappo_shared_ctde_reward_v2",
        },
        "reward": {"version": "reward_v2"},
        "features": {"version": "centralized_critic_context_v1"},
    }

    assert validate_mappo_checkpoint_runner_compatibility(payload, config) == {}


def test_env_compatibility_allows_topology_expansion() -> None:
    payload = _minimal_payload()

    validate_mappo_checkpoint_env_compatibility(
        payload,
        observation_names=[["hour", "load"]] * 6,
        action_names=[["battery"]] * 6,
    )


def test_env_compatibility_rejects_schema_mismatch() -> None:
    payload = _minimal_payload()

    with pytest.raises(ValueError, match="per-building observation schema"):
        validate_mappo_checkpoint_env_compatibility(
            payload,
            observation_names=[["hour", "load", "extra"]] * 6,
            action_names=[["battery"]] * 6,
        )

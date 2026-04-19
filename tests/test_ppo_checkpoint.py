from __future__ import annotations

import pytest

from cos435_citylearn.algorithms.ppo.checkpoints import (
    validate_ppo_checkpoint_env_compatibility,
    validate_ppo_checkpoint_payload_structure,
    validate_ppo_checkpoint_runner_compatibility,
)


def _minimal_payload() -> dict[str, object]:
    return {
        "algorithm": "ppo",
        "control_mode": "shared_dtde",
        "observation_names": [["hour", "load"]] * 3,
        "action_names": [["battery"]] * 3,
        "controller_state": {
            "controller_type": "shared_parameter_ppo",
            "hidden_dimension": [64, 64],
            "lr": 3e-4,
            "clip_range": 0.2,
            "n_epochs": 4,
            "minibatch_size": 16,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "rollout_steps": 32,
            "shared_context_dimension": 4,
            "shared_context_version": "v2",
            "policy_state_dict": {},
            "value_state_dict": {},
            "policy_optimizer_state_dict": {},
            "value_optimizer_state_dict": {},
            "obs_rms_mean": None,
            "obs_rms_var": None,
            "reward_rms_mean": 0.0,
            "reward_rms_var": None,
            "time_step": 0,
        },
    }


def test_structure_validator_accepts_valid_payload() -> None:
    payload = _minimal_payload()
    validate_ppo_checkpoint_payload_structure(payload)


def test_structure_validator_rejects_wrong_algorithm() -> None:
    payload = _minimal_payload()
    payload["algorithm"] = "sac"
    with pytest.raises(ValueError, match="not a PPO checkpoint"):
        validate_ppo_checkpoint_payload_structure(payload)


def test_structure_validator_rejects_centralized_control_mode() -> None:
    payload = _minimal_payload()
    payload["control_mode"] = "centralized"
    with pytest.raises(ValueError, match="shared PPO only supports control_mode=shared_dtde"):
        validate_ppo_checkpoint_payload_structure(payload)


def test_structure_validator_rejects_v1_context() -> None:
    payload = _minimal_payload()
    payload["controller_state"]["shared_context_version"] = "v1"
    with pytest.raises(ValueError, match="v2"):
        validate_ppo_checkpoint_payload_structure(payload)


def test_runner_compatibility_enforces_algorithm() -> None:
    payload = _minimal_payload()
    config = {"algorithm": {"name": "sac", "control_mode": "shared_dtde"}}
    with pytest.raises(ValueError, match="incompatible"):
        validate_ppo_checkpoint_runner_compatibility(payload, config)


def test_env_compatibility_allows_3_to_6_topology_expansion() -> None:
    payload = _minimal_payload()
    payload["observation_names"] = [["hour", "load"]] * 3
    payload["action_names"] = [["battery"]] * 3
    env_obs = [["hour", "load"]] * 6
    env_act = [["battery"]] * 6
    validate_ppo_checkpoint_env_compatibility(
        payload, observation_names=env_obs, action_names=env_act
    )


def test_env_compatibility_rejects_per_building_schema_mismatch() -> None:
    payload = _minimal_payload()
    payload["observation_names"] = [["hour", "load"]] * 3
    env_obs = [["hour", "load", "new_feature"]] * 6
    env_act = [["battery"]] * 6
    with pytest.raises(ValueError, match="per-building observation schema"):
        validate_ppo_checkpoint_env_compatibility(
            payload, observation_names=env_obs, action_names=env_act
        )


def test_env_compatibility_rejects_action_schema_mismatch() -> None:
    payload = _minimal_payload()
    payload["action_names"] = [["battery"]] * 3
    env_obs = [["hour", "load"]] * 6
    env_act = [["battery", "heat_pump"]] * 6
    with pytest.raises(ValueError, match="action schema"):
        validate_ppo_checkpoint_env_compatibility(
            payload, observation_names=env_obs, action_names=env_act
        )


def test_env_compatibility_rejects_non_first_building_action_drift() -> None:
    # checkpoint buildings consistent, env building 2 differs -> must catch
    payload = _minimal_payload()
    payload["action_names"] = [["battery"]] * 3
    env_obs = [["hour", "load"]] * 6
    env_act = [["battery"], ["battery"], ["heat_pump"], ["battery"], ["battery"], ["battery"]]
    with pytest.raises(ValueError, match="inconsistent per-building action schemas"):
        validate_ppo_checkpoint_env_compatibility(
            payload, observation_names=env_obs, action_names=env_act
        )


def test_env_compatibility_rejects_checkpoint_action_inconsistency() -> None:
    # checkpoint itself has drift across buildings
    payload = _minimal_payload()
    payload["action_names"] = [["battery"], ["heat_pump"], ["battery"]]
    env_obs = [["hour", "load"]] * 6
    env_act = [["battery"]] * 6
    with pytest.raises(ValueError, match="inconsistent per-building action schemas"):
        validate_ppo_checkpoint_env_compatibility(
            payload, observation_names=env_obs, action_names=env_act
        )

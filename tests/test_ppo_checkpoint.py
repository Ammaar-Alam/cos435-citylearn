from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from cos435_citylearn.algorithms.ppo.checkpoints import (
    validate_ppo_checkpoint_env_compatibility,
    validate_ppo_checkpoint_payload_structure,
    validate_ppo_checkpoint_runner_compatibility,
)
from cos435_citylearn.baselines.ppo import _build_shared_ppo_checkpoint_payload


def _minimal_payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "algorithm": "ppo",
        "control_mode": "shared_dtde",
        "variant": "shared_dtde_reward_v2",
        "reward_version": "v2",
        "features_version": "v2",
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
            "max_grad_norm": 0.5,
            "rollout_steps": 32,
            "reward_scaling": 1.0,
            "shared_context_dimension": 4,
            "shared_context_version": "v2",
            "normalize_observations": True,
            "normalize_rewards": True,
            "target_kl": None,
            "policy_state_dict": {},
            "value_state_dict": {},
            "policy_optimizer_state_dict": {},
            "value_optimizer_state_dict": {},
            "obs_rms_mean": None,
            "obs_rms_var": None,
            "obs_rms_count": 0.0,
            "reward_rms_mean": 0.0,
            "reward_rms_var": None,
            "reward_rms_count": 0.0,
            "time_step": 0,
            "total_updates": 0,
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


def test_structure_validator_rejects_unknown_format_version() -> None:
    payload = _minimal_payload()
    payload["format_version"] = 99
    with pytest.raises(ValueError, match="format_version"):
        validate_ppo_checkpoint_payload_structure(payload)


def test_runner_compatibility_enforces_algorithm() -> None:
    payload = _minimal_payload()
    config = {"algorithm": {"name": "sac", "control_mode": "shared_dtde"}}
    with pytest.raises(ValueError, match="incompatible"):
        validate_ppo_checkpoint_runner_compatibility(payload, config)


def _matching_runner_config() -> dict[str, object]:
    return {
        "algorithm": {
            "name": "ppo",
            "control_mode": "shared_dtde",
            "variant": "shared_dtde_reward_v2",
        },
        "reward": {"version": "v2"},
        "features": {"version": "v2"},
    }


def test_runner_compatibility_passes_on_matching_runtime_labels() -> None:
    payload = _minimal_payload()
    config = _matching_runner_config()
    mismatches = validate_ppo_checkpoint_runner_compatibility(payload, config)
    assert mismatches == {}


def test_runner_compatibility_rejects_reward_mismatch_without_flag() -> None:
    payload = _minimal_payload()
    config = _matching_runner_config()
    config["reward"]["version"] = "v1"
    with pytest.raises(ValueError, match="reward_version"):
        validate_ppo_checkpoint_runner_compatibility(payload, config)


def test_runner_compatibility_allows_reward_mismatch_with_flag(caplog) -> None:
    payload = _minimal_payload()
    config = _matching_runner_config()
    config["reward"]["version"] = "v1"
    config["features"]["version"] = "v1"
    with caplog.at_level(logging.WARNING, logger="cos435_citylearn.algorithms.ppo.checkpoints"):
        mismatches = validate_ppo_checkpoint_runner_compatibility(
            payload, config, allow_cross_reward_eval=True
        )
    assert mismatches == {
        "reward_version": ("v2", "v1"),
        "features_version": ("v2", "v1"),
    }
    assert "allow_cross_reward_eval=True" in caplog.text


def test_runner_compatibility_falls_back_to_nested_config_for_legacy_payload() -> None:
    # pre-P1.2 checkpoints didn't write variant/reward_version/features_version at
    # the payload's top level; they only carried them nested inside the config
    # snapshot. the tightened validator must still accept those as matching.
    payload = _minimal_payload()
    del payload["variant"]
    del payload["reward_version"]
    del payload["features_version"]
    payload["config"] = {
        "algorithm": {"variant": "shared_dtde_reward_v2"},
        "reward": {"version": "v2"},
        "features": {"version": "v2"},
    }
    config = _matching_runner_config()
    mismatches = validate_ppo_checkpoint_runner_compatibility(payload, config)
    assert mismatches == {}


def test_runner_compatibility_detects_legacy_payload_mismatch() -> None:
    # fallback still catches genuine label drift when only the nested config is
    # available (ensures we didn't accidentally make legacy payloads a silent
    # bypass).
    payload = _minimal_payload()
    del payload["variant"]
    del payload["reward_version"]
    del payload["features_version"]
    payload["config"] = {
        "algorithm": {"variant": "shared_dtde_reward_v2"},
        "reward": {"version": "v1"},  # legacy payload trained on v1
        "features": {"version": "v2"},
    }
    config = _matching_runner_config()  # runner wants reward v2
    with pytest.raises(ValueError, match="reward_version"):
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


def test_shared_ppo_payload_carries_top_level_runtime_labels() -> None:
    # Guard against regressing variant/reward_version/features_version back into
    # config-only storage. The runner-compat validator reads top-level first
    # (see H2's _trained_runtime_labels) so if this payload stopped including
    # them at top level, the validator would silently fall back and mask label
    # drift introduced mid-training.
    config = {
        "algorithm": {
            "name": "ppo",
            "control_mode": "shared_dtde",
            "variant": "shared_dtde_reward_v2",
        },
        "reward": {"version": "v2"},
        "features": {"version": "v2"},
        "training": {"seed": 7},
    }
    env_bundle = SimpleNamespace(
        dataset_name="citylearn_challenge_2023",
        schema_path="schema.json",
        env=SimpleNamespace(
            observation_names=[["hour", "load"]],
            action_names=[["battery"]],
        ),
    )
    controller = SimpleNamespace(
        checkpoint_state=lambda: {"controller_type": "shared_parameter_ppo"}
    )

    payload = _build_shared_ppo_checkpoint_payload(
        run_id="ppo__shared_dtde_reward_v2__public_dev__seed7__20260419T000000Z__abcd1234",
        config=config,
        env_bundle=env_bundle,
        controller=controller,
        training_step=256,
    )

    assert payload["variant"] == "shared_dtde_reward_v2"
    assert payload["reward_version"] == "v2"
    assert payload["features_version"] == "v2"
    assert payload["algorithm"] == "ppo"
    assert payload["control_mode"] == "shared_dtde"

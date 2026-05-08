from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest

from cos435_citylearn.algorithms.td3.checkpoints import (
    safe_load_td3_checkpoint_payload,
    validate_td3_checkpoint_env_compatibility,
    validate_td3_checkpoint_payload_structure,
    validate_td3_checkpoint_runner_compatibility,
)
from cos435_citylearn.algorithms.td3.controllers import (
    SharedTD3Controller,
    _restore_time_state,
)


def _minimal_shared_td3_payload() -> dict[str, object]:
    return {
        "algorithm": "td3",
        "control_mode": "shared_dtde",
        "variant": "td3_shared_dtde_reward_v2",
        "reward_version": "reward_v2",
        "features_version": "shared_district_context_v2",
        "observation_names": [["hour", "load"], ["hour", "load"], ["hour", "load"]],
        "action_names": [["battery"], ["battery"], ["battery"]],
        "controller_state": {
            "controller_type": "shared_parameter_td3",
            "hidden_dimension": [64, 64],
            "discount": 0.99,
            "tau": 0.005,
            "lr": 0.0003,
            "batch_size": 16,
            "replay_buffer_capacity": 128,
            "standardize_start_time_step": 8,
            "end_exploration_time_step": 8,
            "action_scaling_coefficient": 0.5,
            "reward_scaling": 5.0,
            "update_per_time_step": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "exploration_noise": 0.1,
            "time_step": 0,
            "total_updates": 0,
            "normalized": False,
            "actor_state_dict": {},
            "critic1_state_dict": {},
            "critic2_state_dict": {},
            "target_actor_state_dict": {},
            "target_critic1_state_dict": {},
            "target_critic2_state_dict": {},
            "actor_optimizer_state_dict": {},
            "critic1_optimizer_state_dict": {},
            "critic2_optimizer_state_dict": {},
            "norm_mean": None,
            "norm_std": None,
            "r_norm_mean": None,
            "r_norm_std": None,
            "shared_context_dimension": 4,
            "shared_context_version": "v2",
        },
    }


def _matching_runner_config() -> dict[str, object]:
    return {
        "algorithm": {
            "name": "td3",
            "control_mode": "shared_dtde",
            "variant": "td3_shared_dtde_reward_v2",
        },
        "reward": {"version": "reward_v2"},
        "features": {"version": "shared_district_context_v2"},
    }


def test_restore_time_state_resets_citylearn_private_state() -> None:
    class DummyController:
        action_space = [object(), object()]

    controller = DummyController()

    _restore_time_state(controller, restored_time_step=3)

    assert controller._Environment__time_step == 3
    assert controller._Agent__actions == [[[], [], [], []], [[], [], [], []]]


def test_shared_td3_checkpoint_load_restores_time_step() -> None:
    source = inspect.getsource(SharedTD3Controller.load_checkpoint_state)
    assert '_restore_time_state(self, restored_time_step=int(payload["time_step"]))' in source


def test_safe_load_td3_checkpoint_payload_uses_weights_only(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")
    payload = _minimal_shared_td3_payload()

    def fake_load(path, *, map_location, weights_only):
        assert Path(path) == checkpoint_path
        assert map_location == "cpu"
        assert weights_only is True
        return payload

    monkeypatch.setattr(
        "cos435_citylearn.algorithms.td3.checkpoints.torch.load",
        fake_load,
    )

    loaded = safe_load_td3_checkpoint_payload(checkpoint_path)
    assert loaded == payload


def test_validate_td3_checkpoint_payload_structure_rejects_wrong_algorithm() -> None:
    payload = _minimal_shared_td3_payload()
    payload["algorithm"] = "sac"

    with pytest.raises(ValueError, match="not a TD3 checkpoint"):
        validate_td3_checkpoint_payload_structure(payload)


def test_validate_td3_checkpoint_runner_compatibility_passes_matching_labels() -> None:
    payload = _minimal_shared_td3_payload()
    mismatches = validate_td3_checkpoint_runner_compatibility(payload, _matching_runner_config())
    assert mismatches == {}


def test_validate_td3_checkpoint_runner_compatibility_rejects_reward_mismatch() -> None:
    payload = _minimal_shared_td3_payload()
    config = _matching_runner_config()
    config["reward"]["version"] = "reward_v1"

    with pytest.raises(ValueError, match="reward_version"):
        validate_td3_checkpoint_runner_compatibility(payload, config)


def test_validate_td3_checkpoint_runner_compatibility_allows_label_mismatch(caplog) -> None:
    payload = _minimal_shared_td3_payload()
    config = _matching_runner_config()
    config["features"]["version"] = "shared_district_context"

    with caplog.at_level(logging.WARNING, logger="cos435_citylearn.algorithms.td3.checkpoints"):
        mismatches = validate_td3_checkpoint_runner_compatibility(
            payload,
            config,
            allow_cross_reward_eval=True,
        )

    assert mismatches == {
        "features_version": ("shared_district_context_v2", "shared_district_context")
    }
    assert "allow_cross_reward_eval=True" in caplog.text


def test_validate_td3_checkpoint_env_compatibility_allows_3_to_6_transfer() -> None:
    payload = _minimal_shared_td3_payload()
    validate_td3_checkpoint_env_compatibility(
        payload,
        observation_names=[["hour", "load"]] * 6,
        action_names=[["battery"]] * 6,
    )


def test_validate_td3_checkpoint_env_compatibility_rejects_per_building_mismatch() -> None:
    payload = _minimal_shared_td3_payload()

    with pytest.raises(ValueError, match="observation schema"):
        validate_td3_checkpoint_env_compatibility(
            payload,
            observation_names=[["hour", "load", "extra"]] * 6,
            action_names=[["battery"]] * 6,
        )

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from cos435_citylearn.algorithms.sac import resolve_reward_function
from cos435_citylearn.algorithms.sac.checkpoints import (
    safe_load_checkpoint_payload,
    validate_checkpoint_env_compatibility,
    validate_checkpoint_payload_structure,
    validate_checkpoint_runner_compatibility,
)
from cos435_citylearn.baselines.sac import (
    _build_adapter,
    _build_checkpoint_payload,
    _instantiate_controller,
    _instantiate_controller_from_checkpoint,
    _load_imported_checkpoint,
    _maybe_reset_reward,
)
from cos435_citylearn.config import load_yaml
from cos435_citylearn.env import make_citylearn_env
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


def _minimal_central_checkpoint_payload() -> dict[str, object]:
    return {
        "algorithm": "sac",
        "control_mode": "centralized",
        "variant": "centralized_baseline",
        "reward_version": "v2",
        "features_version": "v2",
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
            "time_step": 0,
            "normalized": [False],
            "policy_state_dicts": [{}],
            "soft_q1_state_dicts": [{}],
            "soft_q2_state_dicts": [{}],
            "target_soft_q1_state_dicts": [{}],
            "target_soft_q2_state_dicts": [{}],
            "policy_optimizer_state_dicts": [{}],
            "soft_q_optimizer1_state_dicts": [{}],
            "soft_q_optimizer2_state_dicts": [{}],
            "log_alpha": [None],
            "alpha_optimizer_state_dicts": [None],
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
        "variant": "shared_dtde_reward_v2",
        "reward_version": "v2",
        "features_version": "v2",
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
            "time_step": 0,
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


def _matching_runner_config_shared() -> dict[str, object]:
    return {
        "algorithm": {
            "name": "sac",
            "control_mode": "shared_dtde",
            "variant": "shared_dtde_reward_v2",
        },
        "reward": {"version": "v2"},
        "features": {"version": "v2"},
    }


def test_validate_checkpoint_runner_compatibility_passes_on_matching_runtime_labels() -> None:
    payload = _minimal_shared_checkpoint_payload()
    config = _matching_runner_config_shared()
    mismatches = validate_checkpoint_runner_compatibility(payload, config)
    assert mismatches == {}


def test_validate_checkpoint_runner_compatibility_rejects_reward_mismatch_without_flag() -> None:
    payload = _minimal_shared_checkpoint_payload()
    config = _matching_runner_config_shared()
    config["reward"]["version"] = "v1"
    with pytest.raises(ValueError, match="reward_version"):
        validate_checkpoint_runner_compatibility(payload, config)


def test_validate_checkpoint_runner_compatibility_allows_reward_mismatch_with_flag(caplog) -> None:
    payload = _minimal_shared_checkpoint_payload()
    config = _matching_runner_config_shared()
    config["reward"]["version"] = "v1"
    config["features"]["version"] = "v1"
    with caplog.at_level(logging.WARNING, logger="cos435_citylearn.algorithms.sac.checkpoints"):
        mismatches = validate_checkpoint_runner_compatibility(
            payload, config, allow_cross_reward_eval=True
        )
    assert mismatches == {
        "reward_version": ("v2", "v1"),
        "features_version": ("v2", "v1"),
    }
    assert "allow_cross_reward_eval=True" in caplog.text


def test_validate_checkpoint_runner_compatibility_falls_back_to_nested_config() -> None:
    # pre-P1.2 SAC checkpoints only carry variant/reward_version/features_version
    # inside the nested config snapshot. the tightened validator must still accept
    # those as matching (otherwise the Codex fix regresses every older run).
    payload = _minimal_shared_checkpoint_payload()
    del payload["variant"]
    del payload["reward_version"]
    del payload["features_version"]
    payload["config"] = {
        "algorithm": {"variant": "shared_dtde_reward_v2"},
        "reward": {"version": "v2"},
        "features": {"version": "v2"},
    }
    config = _matching_runner_config_shared()
    mismatches = validate_checkpoint_runner_compatibility(payload, config)
    assert mismatches == {}


def test_validate_checkpoint_runner_compatibility_detects_legacy_payload_mismatch() -> None:
    # fallback still catches genuine label drift -- this guards against the
    # fallback turning into a silent bypass for mismatched checkpoints.
    payload = _minimal_shared_checkpoint_payload()
    del payload["variant"]
    del payload["reward_version"]
    del payload["features_version"]
    payload["config"] = {
        "algorithm": {"variant": "shared_dtde_reward_v2"},
        "reward": {"version": "v1"},
        "features": {"version": "v2"},
    }
    config = _matching_runner_config_shared()  # expects reward v2
    with pytest.raises(ValueError, match="reward_version"):
        validate_checkpoint_runner_compatibility(payload, config)


def test_load_imported_checkpoint_accepts_artifacts_root_only(tmp_path: Path) -> None:
    # Codex P1 (2026-04-20): SAC eval with artifact_id + artifacts_root but no
    # imported_artifacts_root must resolve against artifacts_root (tests / batch
    # jobs that point artifact_id at a custom local run root). Previously this
    # combination raised ValueError, breaking re-eval workflows that the PPO
    # path already supported.
    artifact_id = "sac__smoke__public_dev__seed0"
    run_dir = tmp_path / "runs" / artifact_id
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    payload = _minimal_shared_checkpoint_payload()
    torch.save(payload, checkpoint_path)

    resolved_path, loaded_payload = _load_imported_checkpoint(
        artifact_id=artifact_id,
        imported_artifacts_root=None,
        artifacts_root=tmp_path,
    )

    assert resolved_path == checkpoint_path
    assert loaded_payload["algorithm"] == "sac"
    assert loaded_payload["control_mode"] == "shared_dtde"


def test_load_imported_checkpoint_requires_artifacts_root_when_importing(tmp_path: Path) -> None:
    # With imported_artifacts_root set but artifacts_root missing we cannot
    # resolve relative file_path entries in the artifact record -- fail loudly
    # rather than silently assuming a results_root default.
    with pytest.raises(ValueError, match="artifacts_root must be set"):
        _load_imported_checkpoint(
            artifact_id="anything",
            imported_artifacts_root=tmp_path / "imported",
            artifacts_root=None,
        )


def test_load_imported_checkpoint_reports_missing_local_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SAC checkpoint not found"):
        _load_imported_checkpoint(
            artifact_id="missing_run",
            imported_artifacts_root=None,
            artifacts_root=tmp_path,
        )


def test_validate_checkpoint_env_compatibility_rejects_schema_mismatch() -> None:
    payload = _minimal_central_checkpoint_payload()

    with pytest.raises(ValueError, match="observation schema"):
        validate_checkpoint_env_compatibility(
            payload,
            observation_names=[["hour", "different_load"]],
            action_names=[["battery"]],
        )


def test_validate_checkpoint_env_compatibility_rejects_building_count_mismatch_for_central():
    """Central checkpoints must fail loudly when eval env has more buildings."""
    payload = _minimal_central_checkpoint_payload()  # 1-building checkpoint

    with pytest.raises(ValueError, match="trained on 1 buildings but target env has 2"):
        validate_checkpoint_env_compatibility(
            payload,
            observation_names=[["hour", "load"], ["hour", "load"]],
            action_names=[["battery"], ["battery"]],
        )


def test_validate_checkpoint_env_compatibility_allows_shared_building_count_change() -> None:
    """Shared checkpoints are topology-invariant: 1->3 buildings must pass."""
    payload = _minimal_shared_checkpoint_payload()  # 1-building checkpoint

    validate_checkpoint_env_compatibility(
        payload,
        observation_names=[["hour", "load"], ["hour", "load"], ["hour", "load"]],
        action_names=[["battery"], ["battery"], ["battery"]],
    )


def test_validate_shared_checkpoint_env_compatibility_allows_more_buildings_with_matching_schema() -> None:
    """Shared checkpoints must pass when moving from 1 to 6 buildings (phase_3 size)."""
    payload = _minimal_shared_checkpoint_payload()

    validate_checkpoint_env_compatibility(
        payload,
        observation_names=[["hour", "load"]] * 6,
        action_names=[["battery"]] * 6,
    )


def test_validate_checkpoint_env_compatibility_rejects_shared_per_building_mismatch() -> None:
    """Shared checkpoints must still fail when per-building schemas differ."""
    payload = _minimal_shared_checkpoint_payload()

    with pytest.raises(ValueError, match="per-building observation schema"):
        validate_checkpoint_env_compatibility(
            payload,
            observation_names=[["hour", "solar"]],  # different feature set
            action_names=[["battery"]],
        )


def test_shared_checkpoint_rejects_non_first_building_action_drift() -> None:
    # env buildings[2] diverges -> previous code compared only [0] so would pass silently
    payload = _minimal_shared_checkpoint_payload()
    payload["action_names"] = [["battery"]]
    env_obs = [["hour", "load"]] * 6
    env_act = [["battery"], ["battery"], ["heat_pump"], ["battery"], ["battery"], ["battery"]]

    with pytest.raises(ValueError, match="inconsistent per-building action schemas"):
        validate_checkpoint_env_compatibility(
            payload, observation_names=env_obs, action_names=env_act
        )



def test_validate_checkpoint_payload_structure_rejects_missing_shared_context_dimension() -> None:
    payload = _minimal_shared_checkpoint_payload()
    del payload["controller_state"]["shared_context_dimension"]

    with pytest.raises(ValueError, match="shared_context_dimension"):
        validate_checkpoint_payload_structure(payload)


def test_validate_checkpoint_payload_structure_rejects_missing_entropy_fields() -> None:
    payload = _minimal_central_checkpoint_payload()
    del payload["controller_state"]["log_alpha"]

    with pytest.raises(ValueError, match="log_alpha"):
        validate_checkpoint_payload_structure(payload)


def test_validate_checkpoint_payload_structure_rejects_missing_time_step() -> None:
    payload = _minimal_shared_checkpoint_payload()
    del payload["controller_state"]["time_step"]

    with pytest.raises(ValueError, match="time_step"):
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


def test_shared_checkpoint_roundtrip_restores_time_step(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    config, observations, _reference_actions, checkpoint_path = _warm_checkpoint_payload(
        "configs/train/sac/sac_shared_dtde_smoke.yaml", tmp_path
    )
    loaded_payload = torch.load(checkpoint_path, map_location="cpu")
    reward_function = resolve_reward_function(config["reward"]["version"])
    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=False,
        reward_function=reward_function,
    )

    reloaded_controller = _instantiate_controller_from_checkpoint(
        eval_env_bundle.env,
        loaded_payload,
    )

    restored_time_step = loaded_payload["controller_state"]["time_step"]
    assert reloaded_controller.time_step == restored_time_step
    assert reloaded_controller.time_step > reloaded_controller.end_exploration_time_step

    actions = reloaded_controller.predict(observations, deterministic=False)

    assert len(actions) == len(observations)
    assert reloaded_controller.time_step == restored_time_step + 1


def test_centralized_checkpoint_roundtrip_restores_time_step(tmp_path: Path) -> None:
    require_benchmark_runtime()
    require_dataset()
    config, observations, _reference_actions, checkpoint_path = _warm_checkpoint_payload(
        "configs/train/sac/sac_central_smoke.yaml", tmp_path
    )
    loaded_payload = torch.load(checkpoint_path, map_location="cpu")
    reward_function = resolve_reward_function(config["reward"]["version"])
    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=True,
        reward_function=reward_function,
    )

    reloaded_controller = _instantiate_controller_from_checkpoint(
        eval_env_bundle.env,
        loaded_payload,
    )

    restored_time_step = loaded_payload["controller_state"]["time_step"]
    assert reloaded_controller.time_step == restored_time_step
    assert reloaded_controller.time_step > reloaded_controller.end_exploration_time_step

    actions = reloaded_controller.predict(observations, deterministic=False)

    assert len(actions) == len(observations)
    assert reloaded_controller.time_step == restored_time_step + 1

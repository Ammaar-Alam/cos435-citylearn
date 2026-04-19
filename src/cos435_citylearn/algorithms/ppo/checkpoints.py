from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from cos435_citylearn.algorithms.ppo.shared_features import SHARED_CONTEXT_V2_DIMENSION

SUPPORTED_FORMAT_VERSIONS = {1}

REQUIRED_CHECKPOINT_KEYS = {
    "format_version",
    "algorithm",
    "control_mode",
    "controller_state",
    "observation_names",
    "action_names",
}

COMMON_CONTROLLER_STATE_KEYS = {
    "controller_type",
    "hidden_dimension",
    "lr",
    "clip_range",
    "n_epochs",
    "minibatch_size",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "rollout_steps",
    "reward_scaling",
    "shared_context_dimension",
    "shared_context_version",
    "normalize_observations",
    "normalize_rewards",
    "target_kl",
    "policy_state_dict",
    "value_state_dict",
    "policy_optimizer_state_dict",
    "value_optimizer_state_dict",
    "obs_rms_mean",
    "obs_rms_var",
    "obs_rms_count",
    "reward_rms_mean",
    "reward_rms_var",
    "reward_rms_count",
    "time_step",
    "total_updates",
}


def safe_load_ppo_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"failed to safely load PPO checkpoint: {exc}") from exc

    validate_ppo_checkpoint_payload_structure(payload)
    return payload


def validate_ppo_checkpoint_payload_structure(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("PPO checkpoint payload must be a mapping")

    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(payload))
    if missing:
        raise ValueError(
            f"PPO checkpoint payload missing required keys: {', '.join(missing)}"
        )

    format_version = payload["format_version"]
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported PPO checkpoint format_version: {format_version}. "
            f"supported versions: {sorted(SUPPORTED_FORMAT_VERSIONS)}"
        )

    if payload["algorithm"] != "ppo":
        raise ValueError("checkpoint payload is not a PPO checkpoint")

    if payload["control_mode"] != "shared_dtde":
        raise ValueError(
            f"unsupported PPO checkpoint control_mode: {payload['control_mode']}. "
            "shared PPO only supports control_mode=shared_dtde."
        )

    if not isinstance(payload["observation_names"], list) or not isinstance(
        payload["action_names"], list
    ):
        raise ValueError("PPO checkpoint schema metadata is malformed")

    controller_state = payload["controller_state"]
    if not isinstance(controller_state, dict):
        raise ValueError("PPO checkpoint controller_state must be a mapping")

    missing_state = sorted(COMMON_CONTROLLER_STATE_KEYS - set(controller_state))
    if missing_state:
        raise ValueError(
            "PPO checkpoint controller_state missing required keys: " + ", ".join(missing_state)
        )

    controller_type = controller_state["controller_type"]
    if controller_type != "shared_parameter_ppo":
        raise ValueError(f"unknown PPO checkpoint controller type: {controller_type}")

    try:
        shared_context_dimension = int(controller_state["shared_context_dimension"])
    except (TypeError, ValueError) as exc:
        raise ValueError("PPO checkpoint shared_context_dimension must be an integer") from exc

    if shared_context_dimension != SHARED_CONTEXT_V2_DIMENSION:
        raise ValueError(
            f"PPO checkpoint shared_context_dimension must be {SHARED_CONTEXT_V2_DIMENSION}"
        )

    if controller_state.get("shared_context_version") != "v2":
        raise ValueError("PPO checkpoint shared_context_version must be 'v2'")


def validate_ppo_checkpoint_runner_compatibility(
    checkpoint_payload: dict[str, Any],
    config: dict[str, Any],
) -> None:
    expected_algorithm = str(config["algorithm"]["name"])
    expected_control_mode = str(config["algorithm"]["control_mode"])

    if checkpoint_payload["algorithm"] != expected_algorithm:
        raise ValueError(
            f"checkpoint algorithm '{checkpoint_payload['algorithm']}' is "
            f"incompatible with runner algorithm '{expected_algorithm}'"
        )

    if checkpoint_payload["control_mode"] != expected_control_mode:
        raise ValueError(
            f"checkpoint control_mode '{checkpoint_payload['control_mode']}' is "
            f"incompatible with runner control_mode '{expected_control_mode}'"
        )


def validate_ppo_checkpoint_env_compatibility(
    checkpoint_payload: dict[str, Any],
    *,
    observation_names: Sequence[Sequence[str]],
    action_names: Sequence[Sequence[str]],
) -> None:
    # building count can differ (3 -> 6) but per-building schema must match
    checkpoint_observation_names = checkpoint_payload["observation_names"]
    checkpoint_action_names = checkpoint_payload["action_names"]
    env_observation_names = list(observation_names)
    env_action_names = list(action_names)

    if not checkpoint_observation_names or not env_observation_names:
        raise ValueError(
            "shared PPO checkpoint requires non-empty observation schema on both sides"
        )

    ckpt_per_building_obs = checkpoint_observation_names[0]
    env_per_building_obs = env_observation_names[0]
    if any(list(b) != ckpt_per_building_obs for b in checkpoint_observation_names):
        raise ValueError("shared PPO checkpoint has inconsistent per-building observation schemas")
    if any(list(b) != env_per_building_obs for b in env_observation_names):
        raise ValueError(
            "target env has inconsistent per-building observation schemas; "
            "shared PPO requires identical buildings"
        )
    if list(ckpt_per_building_obs) != list(env_per_building_obs):
        raise ValueError(
            "shared PPO checkpoint per-building observation schema does not match target env; "
            "the two datasets expose different building features."
        )

    if not checkpoint_action_names or not env_action_names:
        raise ValueError(
            "shared PPO checkpoint requires non-empty action schema on both sides"
        )
    ckpt_per_building_act = checkpoint_action_names[0]
    env_per_building_act = env_action_names[0]
    if any(list(b) != ckpt_per_building_act for b in checkpoint_action_names):
        raise ValueError("shared PPO checkpoint has inconsistent per-building action schemas")
    if any(list(b) != env_per_building_act for b in env_action_names):
        raise ValueError(
            "target env has inconsistent per-building action schemas; "
            "shared PPO requires identical buildings"
        )
    if list(ckpt_per_building_act) != list(env_per_building_act):
        raise ValueError(
            "shared PPO checkpoint per-building action schema does not match target env."
        )

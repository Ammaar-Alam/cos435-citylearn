from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch

from cos435_citylearn.algorithms._runtime_labels import (
    RUNTIME_LABEL_FIELDS as _RUNTIME_LABEL_FIELDS,
)
from cos435_citylearn.algorithms._runtime_labels import (
    expected_runtime_labels as _expected_runtime_labels,
)
from cos435_citylearn.algorithms._runtime_labels import (
    trained_runtime_labels as _trained_runtime_labels,
)
from cos435_citylearn.algorithms.ppo.shared_features import SHARED_CONTEXT_V2_DIMENSION

_LOGGER = logging.getLogger(__name__)

REQUIRED_CHECKPOINT_KEYS = {
    "algorithm",
    "control_mode",
    "controller_state",
    "observation_names",
    "action_names",
}

SHARED_CONTROLLER_STATE_KEYS = {
    "controller_type",
    "hidden_dimension",
    "discount",
    "tau",
    "lr",
    "batch_size",
    "replay_buffer_capacity",
    "standardize_start_time_step",
    "end_exploration_time_step",
    "action_scaling_coefficient",
    "reward_scaling",
    "update_per_time_step",
    "policy_delay",
    "target_policy_noise",
    "target_noise_clip",
    "exploration_noise",
    "time_step",
    "total_updates",
    "normalized",
    "actor_state_dict",
    "critic1_state_dict",
    "critic2_state_dict",
    "target_actor_state_dict",
    "target_critic1_state_dict",
    "target_critic2_state_dict",
    "actor_optimizer_state_dict",
    "critic1_optimizer_state_dict",
    "critic2_optimizer_state_dict",
    "norm_mean",
    "norm_std",
    "r_norm_mean",
    "r_norm_std",
    "shared_context_dimension",
    "shared_context_version",
}


def safe_load_td3_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch exception varies by version
        raise ValueError(f"failed to safely load TD3 checkpoint: {exc}") from exc

    validate_td3_checkpoint_payload_structure(payload)
    return payload


def validate_td3_checkpoint_payload_structure(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("TD3 checkpoint payload must be a mapping")

    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(payload))
    if missing:
        raise ValueError(f"TD3 checkpoint payload missing required keys: {', '.join(missing)}")

    if payload["algorithm"] != "td3":
        raise ValueError("checkpoint payload is not a TD3 checkpoint")
    if payload["control_mode"] != "shared_dtde":
        raise ValueError("shared TD3 only supports control_mode=shared_dtde")
    if not isinstance(payload["observation_names"], list) or not isinstance(
        payload["action_names"], list
    ):
        raise ValueError("TD3 checkpoint schema metadata is malformed")

    controller_state = payload["controller_state"]
    if not isinstance(controller_state, dict):
        raise ValueError("TD3 checkpoint controller_state must be a mapping")

    missing_state = sorted(SHARED_CONTROLLER_STATE_KEYS - set(controller_state))
    if missing_state:
        raise ValueError(
            "TD3 checkpoint controller_state missing required keys: " + ", ".join(missing_state)
        )

    if controller_state["controller_type"] != "shared_parameter_td3":
        raise ValueError(
            f"unknown TD3 checkpoint controller type: {controller_state['controller_type']}"
        )
    if int(controller_state["shared_context_dimension"]) != SHARED_CONTEXT_V2_DIMENSION:
        raise ValueError(
            f"TD3 checkpoint shared_context_dimension must be {SHARED_CONTEXT_V2_DIMENSION}"
        )
    if str(controller_state["shared_context_version"]) != "v2":
        raise ValueError("TD3 checkpoint shared_context_version must be v2")


def validate_td3_checkpoint_runner_compatibility(
    checkpoint_payload: dict[str, Any],
    config: dict[str, Any],
    *,
    allow_cross_reward_eval: bool = False,
) -> dict[str, tuple[Any, Any]]:
    expected_algorithm = str(config["algorithm"]["name"])
    expected_control_mode = str(config["algorithm"]["control_mode"])

    if checkpoint_payload["algorithm"] != expected_algorithm:
        raise ValueError(
            f"checkpoint algorithm '{checkpoint_payload['algorithm']}' is incompatible "
            f"with runner algorithm '{expected_algorithm}'"
        )
    if checkpoint_payload["control_mode"] != expected_control_mode:
        raise ValueError(
            f"checkpoint control_mode '{checkpoint_payload['control_mode']}' "
            f"is incompatible with runner control_mode '{expected_control_mode}'"
        )

    expected_labels = _expected_runtime_labels(config)
    trained_labels = _trained_runtime_labels(checkpoint_payload)
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in _RUNTIME_LABEL_FIELDS:
        trained = trained_labels.get(field)
        expected = expected_labels.get(field)
        if trained != expected:
            mismatches[field] = (trained, expected)

    if mismatches and not allow_cross_reward_eval:
        parts = [f"{field}: checkpoint={t!r} config={e!r}" for field, (t, e) in mismatches.items()]
        raise ValueError(
            "TD3 checkpoint runtime metadata is incompatible with runner config; "
            "pass allow_cross_reward_eval=True to opt into cross-reward evaluation. "
            + "; ".join(parts)
        )
    if mismatches:
        parts = [f"{field}: checkpoint={t!r} config={e!r}" for field, (t, e) in mismatches.items()]
        _LOGGER.warning(
            "allow_cross_reward_eval=True; evaluating TD3 checkpoint under "
            "mismatched runtime labels: %s",
            "; ".join(parts),
        )
    return mismatches


def validate_td3_checkpoint_env_compatibility(
    checkpoint_payload: dict[str, Any],
    *,
    observation_names: Sequence[Sequence[str]],
    action_names: Sequence[Sequence[str]],
) -> None:
    checkpoint_observation_names = [
        list(names) for names in checkpoint_payload["observation_names"]
    ]
    checkpoint_action_names = [list(names) for names in checkpoint_payload["action_names"]]
    env_observation_names = [list(names) for names in observation_names]
    env_action_names = [list(names) for names in action_names]

    if not checkpoint_observation_names or not env_observation_names:
        raise ValueError("shared TD3 checkpoint requires non-empty observation schema")
    if not checkpoint_action_names or not env_action_names:
        raise ValueError("shared TD3 checkpoint requires non-empty action schema")

    checkpoint_observation_reference = checkpoint_observation_names[0]
    checkpoint_action_reference = checkpoint_action_names[0]
    env_observation_reference = env_observation_names[0]
    env_action_reference = env_action_names[0]

    if any(names != checkpoint_observation_reference for names in checkpoint_observation_names):
        raise ValueError("shared TD3 checkpoint observation schema is internally inconsistent")
    if any(names != checkpoint_action_reference for names in checkpoint_action_names):
        raise ValueError("shared TD3 checkpoint action schema is internally inconsistent")
    if any(names != env_observation_reference for names in env_observation_names):
        raise ValueError("target env observation schema is internally inconsistent")
    if any(names != env_action_reference for names in env_action_names):
        raise ValueError("target env action schema is internally inconsistent")
    if env_observation_reference != checkpoint_observation_reference:
        raise ValueError(
            "shared_dtde checkpoint per-building observation schema does not match target env"
        )
    if env_action_reference != checkpoint_action_reference:
        raise ValueError(
            "shared_dtde checkpoint per-building action schema does not match target env"
        )

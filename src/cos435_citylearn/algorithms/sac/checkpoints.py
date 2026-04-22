from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from cos435_citylearn.algorithms.sac.features import SHARED_CONTEXT_DIMENSION
from cos435_citylearn.paths import REPO_ROOT, RESULTS_DIR

REQUIRED_CHECKPOINT_KEYS = {
    "algorithm",
    "control_mode",
    "controller_state",
    "observation_names",
    "action_names",
}

COMMON_CONTROLLER_STATE_KEYS = {
    "controller_type",
    "hidden_dimension",
    "discount",
    "tau",
    "alpha",
    "lr",
    "batch_size",
    "replay_buffer_capacity",
    "standardize_start_time_step",
    "end_exploration_time_step",
    "action_scaling_coefficient",
    "reward_scaling",
    "update_per_time_step",
    "time_step",
    "normalized",
}

CENTRALIZED_CONTROLLER_STATE_KEYS = {
    "policy_state_dicts",
    "soft_q1_state_dicts",
    "soft_q2_state_dicts",
    "target_soft_q1_state_dicts",
    "target_soft_q2_state_dicts",
    "policy_optimizer_state_dicts",
    "soft_q_optimizer1_state_dicts",
    "soft_q_optimizer2_state_dicts",
    "norm_mean",
    "norm_std",
    "r_norm_mean",
    "r_norm_std",
    "log_alpha",
    "alpha_optimizer_state_dicts",
}

SHARED_CONTROLLER_STATE_KEYS = {
    "policy_state_dict",
    "soft_q1_state_dict",
    "soft_q2_state_dict",
    "target_soft_q1_state_dict",
    "target_soft_q2_state_dict",
    "policy_optimizer_state_dict",
    "soft_q_optimizer1_state_dict",
    "soft_q_optimizer2_state_dict",
    "norm_mean",
    "norm_std",
    "r_norm_mean",
    "r_norm_std",
    "shared_context_dimension",
}


def resolve_imported_checkpoint_path(
    *,
    artifact_id: str,
    imported_artifacts_root: str | Path,
    artifacts_root: str | Path,
) -> Path:
    artifact_record_path = Path(imported_artifacts_root) / artifact_id / "artifact.json"
    if not artifact_record_path.exists():
        raise FileNotFoundError(f"unknown imported artifact: {artifact_id}")

    artifact_record = json.loads(artifact_record_path.read_text())
    candidate = Path(artifact_record["file_path"])
    if candidate.is_absolute():
        return candidate

    artifacts_candidate = Path(artifacts_root) / candidate
    if artifacts_candidate.exists():
        return artifacts_candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    results_candidate = RESULTS_DIR / candidate
    if results_candidate.exists():
        return results_candidate

    raise FileNotFoundError(f"checkpoint file not found for artifact: {artifact_id}")


def safe_load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch exception varies by version
        raise ValueError(f"failed to safely load SAC checkpoint: {exc}") from exc

    validate_checkpoint_payload_structure(payload)
    return payload


def validate_checkpoint_payload_structure(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("SAC checkpoint payload must be a mapping")

    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(payload))
    if missing:
        raise ValueError(
            f"SAC checkpoint payload missing required keys: {', '.join(missing)}"
        )

    if payload["algorithm"] != "sac":
        raise ValueError("checkpoint payload is not a SAC checkpoint")

    if payload["control_mode"] not in {"centralized", "shared_dtde"}:
        raise ValueError(f"unsupported SAC checkpoint control_mode: {payload['control_mode']}")

    if not isinstance(payload["observation_names"], list) or not isinstance(
        payload["action_names"], list
    ):
        raise ValueError("SAC checkpoint schema metadata is malformed")

    controller_state = payload["controller_state"]
    if not isinstance(controller_state, dict):
        raise ValueError("SAC checkpoint controller_state must be a mapping")

    missing_common = sorted(COMMON_CONTROLLER_STATE_KEYS - set(controller_state))
    if missing_common:
        raise ValueError(
            "SAC checkpoint controller_state missing required keys: "
            + ", ".join(missing_common)
        )

    controller_type = controller_state["controller_type"]
    if controller_type == "centralized_native":
        missing_specific = sorted(CENTRALIZED_CONTROLLER_STATE_KEYS - set(controller_state))
    elif controller_type == "shared_parameter_sac":
        missing_specific = sorted(SHARED_CONTROLLER_STATE_KEYS - set(controller_state))
    else:
        raise ValueError(f"unknown SAC checkpoint controller type: {controller_type}")

    if missing_specific:
        raise ValueError(
            "SAC checkpoint controller_state missing required keys: "
            + ", ".join(missing_specific)
        )

    if controller_type == "shared_parameter_sac":
        try:
            shared_context_dimension = int(controller_state["shared_context_dimension"])
        except (TypeError, ValueError) as exc:
            raise ValueError("SAC checkpoint shared_context_dimension must be an integer") from exc

        if shared_context_dimension != SHARED_CONTEXT_DIMENSION:
            raise ValueError(
                f"SAC checkpoint shared_context_dimension must be {SHARED_CONTEXT_DIMENSION}"
            )


def validate_checkpoint_runner_compatibility(
    checkpoint_payload: dict[str, Any],
    config: dict[str, Any],
) -> None:
    expected_algorithm = str(config["algorithm"]["name"])
    expected_control_mode = str(config["algorithm"]["control_mode"])

    if checkpoint_payload["algorithm"] != expected_algorithm:
        raise ValueError(
            f"checkpoint algorithm '{checkpoint_payload['algorithm']}' is incompatible with runner algorithm '{expected_algorithm}'"
        )

    if checkpoint_payload["control_mode"] != expected_control_mode:
        raise ValueError(
            f"checkpoint control_mode '{checkpoint_payload['control_mode']}' is incompatible with runner control_mode '{expected_control_mode}'"
        )


def validate_checkpoint_env_compatibility(
    checkpoint_payload: dict[str, Any],
    *,
    observation_names: Sequence[Sequence[str]],
    action_names: Sequence[Sequence[str]],
) -> None:
    checkpoint_observation_names = [list(names) for names in checkpoint_payload["observation_names"]]
    checkpoint_action_names = [list(names) for names in checkpoint_payload["action_names"]]
    current_observation_names = [list(names) for names in observation_names]
    current_action_names = [list(names) for names in action_names]

    if checkpoint_payload["control_mode"] == "shared_dtde":
        reference_observation_names = checkpoint_observation_names[0]
        reference_action_names = checkpoint_action_names[0]

        if any(names != reference_observation_names for names in checkpoint_observation_names):
            raise ValueError("shared SAC checkpoint observation schema is internally inconsistent")

        if any(names != reference_action_names for names in checkpoint_action_names):
            raise ValueError("shared SAC checkpoint action schema is internally inconsistent")

        if any(names != reference_observation_names for names in current_observation_names):
            raise ValueError(
                "checkpoint observation schema is incompatible with the selected runner config"
            )

        if any(names != reference_action_names for names in current_action_names):
            raise ValueError("checkpoint action schema is incompatible with the selected runner config")

        return

    if current_observation_names != checkpoint_observation_names:
        raise ValueError(
            "checkpoint observation schema is incompatible with the selected runner config"
        )

    if current_action_names != checkpoint_action_names:
        raise ValueError(
            "checkpoint action schema is incompatible with the selected runner config"
        )

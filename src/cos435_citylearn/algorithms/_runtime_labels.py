"""Runtime-label helpers shared between PPO and SAC checkpoint validators.

Both algorithms tag checkpoints with three runtime labels -- ``variant``,
``reward_version``, ``features_version`` -- and hard-fail at eval time when
they disagree with the runner config (see
``validate_ppo_checkpoint_runner_compatibility`` and the SAC equivalent).

The original adversarial-review fix wrote those fields at the checkpoint
payload's top level, but any checkpoint produced before that change only
carries them nested under ``payload["config"]`` (the same config dict the
trainer was launched with). Reading the top level without a fallback would
regress those pre-existing checkpoints from "valid, compatible" to "missing
labels -> mismatch -> hard-fail" even though nothing about the training run
actually changed.

``trained_runtime_labels`` prefers the top-level keys and falls back to the
nested config when they're absent, so the tightened validator stays
backward-compatible with older checkpoints while still recording the label
drift we care about for new runs.
"""

from __future__ import annotations

from typing import Any

# Fields the checkpoint records at train time that must match the eval config
# (or be explicitly opted out via ``allow_cross_reward_eval``).
RUNTIME_LABEL_FIELDS: tuple[str, ...] = ("variant", "reward_version", "features_version")


def expected_runtime_labels(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the runtime labels the runner config *expects* a checkpoint to match."""
    variant = config["algorithm"].get("variant")
    return {
        "variant": None if variant is None else str(variant),
        "reward_version": config.get("reward", {}).get("version"),
        "features_version": config.get("features", {}).get("version"),
    }


def trained_runtime_labels(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve the runtime labels a checkpoint was *trained* under.

    Prefers the top-level payload keys. Falls back to the nested
    ``payload["config"]`` block so pre-P1.2 checkpoints (which only carry the
    labels inside the full config snapshot) still validate cleanly.
    """
    nested_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    nested_algorithm = nested_config.get("algorithm", {}) if isinstance(nested_config, dict) else {}
    nested_reward = nested_config.get("reward", {}) if isinstance(nested_config, dict) else {}
    nested_features = nested_config.get("features", {}) if isinstance(nested_config, dict) else {}

    variant = payload.get("variant")
    if variant is None and isinstance(nested_algorithm, dict):
        variant = nested_algorithm.get("variant")

    reward_version = payload.get("reward_version")
    if reward_version is None and isinstance(nested_reward, dict):
        reward_version = nested_reward.get("version")

    features_version = payload.get("features_version")
    if features_version is None and isinstance(nested_features, dict):
        features_version = nested_features.get("version")

    return {
        "variant": None if variant is None else str(variant),
        "reward_version": reward_version,
        "features_version": features_version,
    }

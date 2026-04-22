from __future__ import annotations

import inspect

import pytest

from cos435_citylearn.baselines import ppo as ppo_module
from cos435_citylearn.baselines.ppo import (
    _build_central_ppo_sidecar,
    _validate_central_ppo_sidecar,
)


def _training_config() -> dict[str, object]:
    return {
        "algorithm": {
            "name": "ppo",
            "control_mode": "centralized",
            "variant": "ppo_central_baseline",
        },
        "reward": {"version": "v2"},
        "features": {"version": "v2"},
        "training": {"seed": 0, "total_timesteps": 2048, "learning_rate": 3e-4},
        "env": {"split": "public_dev", "base_config": "dummy"},
    }


def test_sidecar_captures_runtime_labels_at_top_level() -> None:
    # sidecar must expose variant/reward_version/features_version at the top
    # level so the runner-compatibility check can read them without digging
    # into the nested config snapshot.
    sidecar = _build_central_ppo_sidecar(_training_config())
    assert sidecar["variant"] == "ppo_central_baseline"
    assert sidecar["reward_version"] == "v2"
    assert sidecar["features_version"] == "v2"
    assert sidecar["algorithm"] == "ppo"
    assert sidecar["control_mode"] == "centralized"
    assert sidecar["format_version"] == 1
    # the full config snapshot is kept for forensics / fallback; top-level keys
    # are the authoritative source for validation.
    assert sidecar["config"] == _training_config()


def test_validate_passes_on_matching_sidecar() -> None:
    config = _training_config()
    sidecar = _build_central_ppo_sidecar(config)
    mismatches = _validate_central_ppo_sidecar(
        sidecar, config, allow_cross_reward_eval=False, artifact_id="art-123"
    )
    assert mismatches == {}


def test_validate_rejects_reward_mismatch_without_flag() -> None:
    sidecar = _build_central_ppo_sidecar(_training_config())
    mismatched_config = _training_config()
    mismatched_config["reward"]["version"] = "v1"

    with pytest.raises(ValueError, match="reward_version"):
        _validate_central_ppo_sidecar(
            sidecar, mismatched_config, allow_cross_reward_eval=False, artifact_id="art-123"
        )


def test_validate_allows_reward_mismatch_with_flag(capsys) -> None:
    sidecar = _build_central_ppo_sidecar(_training_config())
    mismatched_config = _training_config()
    mismatched_config["reward"]["version"] = "v1"

    mismatches = _validate_central_ppo_sidecar(
        sidecar, mismatched_config, allow_cross_reward_eval=True, artifact_id="art-123"
    )
    assert mismatches == {"reward_version": ("v2", "v1")}
    assert "mismatched runtime labels" in capsys.readouterr().err


def test_validate_missing_sidecar_hard_fails_by_default() -> None:
    # A pre-sidecar artifact must not silently evaluate under a different
    # reward/feature config -- force the user to opt in.
    with pytest.raises(ValueError, match="missing checkpoint_metadata.json"):
        _validate_central_ppo_sidecar(
            None, _training_config(), allow_cross_reward_eval=False, artifact_id="legacy-art"
        )


def test_validate_missing_sidecar_warns_with_flag(capsys) -> None:
    # Escape hatch for legacy artifacts: explicit opt-in surfaces a warning
    # instead of blowing up.
    mismatches = _validate_central_ppo_sidecar(
        None, _training_config(), allow_cross_reward_eval=True, artifact_id="legacy-art"
    )
    assert mismatches == {}
    assert "no checkpoint_metadata.json" in capsys.readouterr().err


def test_run_ppo_captures_sidecar_mismatches_for_manifest() -> None:
    # Codex P2 (2026-04-22): previously the caller at baselines/ppo.py
    # discarded the validator's return value, so when a user opted into
    # cross-reward eval with allow_cross_reward_eval=True, the resulting
    # manifest.json only recorded artifact_id + trained_on_split -- never
    # the actual (checkpoint, config) label diff. That made same-reward
    # aggregation silently include cross-reward runs. Guard the two
    # invariants this fix relies on:
    #  1) the validator return is assigned to `label_mismatches`
    #  2) the manifest block writes `runtime_label_mismatches` using the
    #     same {checkpoint, config} schema as shared PPO and SAC
    source = inspect.getsource(ppo_module)
    assert "label_mismatches = _validate_central_ppo_sidecar(" in source, (
        "central PPO must capture the sidecar validator's mismatch dict; "
        "dropping the return re-opens the Codex P2 regression."
    )
    assert 'manifest["runtime_label_mismatches"]' in source, (
        "central PPO manifest must record runtime_label_mismatches for "
        "cross-reward eval parity with shared PPO and SAC."
    )
    assert '"checkpoint": trained, "config": expected' in source, (
        "runtime_label_mismatches schema must match shared PPO / SAC "
        "({checkpoint, config}) so downstream aggregation is uniform."
    )


def test_validate_uses_nested_fallback_when_top_level_missing() -> None:
    # Forward-compat: if a future tool strips the top-level keys but keeps the
    # nested config snapshot, validation should still succeed on matching
    # labels (mirrors the torch-checkpoint fallback in H2).
    sidecar = _build_central_ppo_sidecar(_training_config())
    del sidecar["variant"]
    del sidecar["reward_version"]
    del sidecar["features_version"]

    mismatches = _validate_central_ppo_sidecar(
        sidecar, _training_config(), allow_cross_reward_eval=False, artifact_id="art-legacy-fmt"
    )
    assert mismatches == {}

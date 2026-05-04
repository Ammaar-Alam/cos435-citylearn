from __future__ import annotations

from types import SimpleNamespace

import pytest

from cos435_citylearn.baselines.td3 import (
    _build_central_td3_sidecar,
    _validate_central_td3_sidecar,
    _validate_td3_topology,
)


def _training_config() -> dict[str, object]:
    return {
        "algorithm": {
            "name": "td3",
            "control_mode": "centralized",
            "variant": "central_baseline",
        },
        "reward": {"version": "reward_v2"},
        "features": {"version": "base_central_obs"},
        "training": {"seed": 0, "total_timesteps": 2048, "learning_rate": 3e-4},
        "env": {"split": "public_dev", "base_config": "dummy"},
    }


def test_central_td3_sidecar_captures_runtime_labels() -> None:
    sidecar = _build_central_td3_sidecar(_training_config())
    assert sidecar["algorithm"] == "td3"
    assert sidecar["control_mode"] == "centralized"
    assert sidecar["variant"] == "central_baseline"
    assert sidecar["reward_version"] == "reward_v2"
    assert sidecar["features_version"] == "base_central_obs"
    assert sidecar["format_version"] == 1


def test_central_td3_sidecar_rejects_label_mismatch_without_flag() -> None:
    sidecar = _build_central_td3_sidecar(_training_config())
    config = _training_config()
    config["reward"]["version"] = "reward_v1"

    with pytest.raises(ValueError, match="reward_version"):
        _validate_central_td3_sidecar(
            sidecar,
            config,
            allow_cross_reward_eval=False,
            artifact_id="art-123",
        )


def test_central_td3_topology_rejects_phase_3_building_count_mismatch() -> None:
    metadata = {
        "observation_names": [["hour"], ["hour"], ["hour"]],
        "action_names": [["battery"], ["battery"], ["battery"]],
    }
    env = SimpleNamespace(
        observation_names=[["hour"]] * 6,
        action_names=[["battery"]] * 6,
    )

    with pytest.raises(ValueError, match="cannot cross building counts"):
        _validate_td3_topology(metadata, env)

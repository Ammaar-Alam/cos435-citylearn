from pathlib import Path

import pytest
import yaml

from cos435_citylearn.algorithms.ppo.controllers import assert_minibatch_fits_rollout
from cos435_citylearn.baselines.ppo import _validate_ppo_topology
from cos435_citylearn.config import assert_training_allowed_on_split
from cos435_citylearn.paths import CONFIGS_DIR


def test_all_yaml_configs_parse() -> None:
    config_paths = sorted(CONFIGS_DIR.rglob("*.yaml"))
    assert config_paths

    for path in config_paths:
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), f"{path} did not parse into a mapping"


def test_train_configs_have_shared_top_level_shape() -> None:
    required = {"env", "algorithm", "reward", "features", "training", "evaluation", "logging"}
    train_paths = sorted((CONFIGS_DIR / "train").rglob("*.yaml"))

    assert train_paths

    for path in train_paths:
        data = yaml.safe_load(path.read_text())
        assert required.issubset(data.keys()), f"{Path(path).name} is missing train sections"


def test_assert_training_allowed_on_split_rejects_held_out_without_artifact() -> None:
    split = {"split": {"name": "phase_3_1", "held_out": True, "tuning_allowed": False}}
    with pytest.raises(ValueError, match="phase_3_1"):
        assert_training_allowed_on_split(split, artifact_id=None)


def test_assert_training_allowed_on_split_allows_held_out_with_artifact() -> None:
    split = {"split": {"name": "phase_3_1", "held_out": True, "tuning_allowed": False}}
    assert_training_allowed_on_split(split, artifact_id="some_run_id")


def test_assert_training_allowed_on_split_allows_non_held_out() -> None:
    split = {"split": {"name": "public_dev", "held_out": False, "tuning_allowed": True}}
    assert_training_allowed_on_split(split, artifact_id=None)


def test_assert_training_allowed_on_split_allows_held_out_with_checkpoint_path() -> None:
    # Codex P1 (2026-04-22): scripts/eval/run_sac_checkpoint.py evaluates a
    # torch checkpoint directly from disk without an artifact_id; the guard
    # must recognize checkpoint_path as the same "eval-only" signal.
    split = {"split": {"name": "phase_3_3", "held_out": True, "tuning_allowed": False}}
    assert_training_allowed_on_split(
        split, artifact_id=None, checkpoint_path="/tmp/some_checkpoint.pt"
    )


def test_assert_training_allowed_on_split_allows_held_out_with_both_signals() -> None:
    # Providing both is still eval-mode; the two XOR check lives in the
    # caller (baselines/sac.py), not the guard itself.
    split = {"split": {"name": "phase_3_2", "held_out": True, "tuning_allowed": False}}
    assert_training_allowed_on_split(
        split, artifact_id="some_run_id", checkpoint_path="/tmp/some_checkpoint.pt"
    )


def test_assert_training_allowed_on_split_rejects_held_out_with_no_signals() -> None:
    # Explicit checkpoint_path=None keyword must still block training on
    # held-out splits (regression guard for the kwarg default).
    split = {"split": {"name": "phase_3_2", "held_out": True, "tuning_allowed": False}}
    with pytest.raises(ValueError, match="phase_3_2"):
        assert_training_allowed_on_split(split, artifact_id=None, checkpoint_path=None)


class _FakeEnv:
    def __init__(self, observation_names, action_names) -> None:
        self.observation_names = observation_names
        self.action_names = action_names


def test_validate_ppo_topology_rejects_building_count_mismatch() -> None:
    metadata = {
        "observation_names": [["hour", "load"], ["hour", "load"], ["hour", "load"]],
        "action_names": [["battery"], ["battery"], ["battery"]],
    }
    env = _FakeEnv(
        observation_names=[["hour", "load"]] * 6,
        action_names=[["battery"]] * 6,
    )

    with pytest.raises(ValueError, match="trained on 3 buildings but target env has 6"):
        _validate_ppo_topology(metadata, env)


def test_validate_ppo_topology_accepts_matching_topology() -> None:
    metadata = {
        "observation_names": [["hour", "load"]] * 3,
        "action_names": [["battery"]] * 3,
    }
    env = _FakeEnv(
        observation_names=[["hour", "load"]] * 3,
        action_names=[["battery"]] * 3,
    )

    _validate_ppo_topology(metadata, env)


def test_minibatch_size_assertion_rejects_oversampling_config() -> None:
    # minibatch 256 exceeds rollout_steps 32 * n_buildings 3 = 96
    with pytest.raises(ValueError, match="minibatch_size"):
        assert_minibatch_fits_rollout(minibatch_size=256, rollout_steps=32, n_buildings=3)


def test_minibatch_size_assertion_allows_fitting_config() -> None:
    # 64 <= 2048 * 3 = 6144
    assert_minibatch_fits_rollout(minibatch_size=64, rollout_steps=2048, n_buildings=3)

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from cos435_citylearn.algorithms.mappo import CentralizedMAPPOController


class _FakeEpisodeTracker:
    def reset(self) -> None:
        return None

    def next_episode(self) -> None:
        return None


class _FakeEnv:
    observation_names = [
        ["hour", "net_electricity_consumption", "electrical_storage_soc", "power_outage"]
    ] * 3
    action_names = [["electrical_storage"]] * 3
    observation_space = [
        spaces.Box(
            low=np.array([0.0, -10.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([24.0, 10.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        for _ in range(3)
    ]
    action_space = [
        spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        for _ in range(3)
    ]
    time_steps = 10
    seconds_per_time_step = 3600
    random_seed = 0
    episode_tracker = _FakeEpisodeTracker()

    def get_metadata(self) -> dict[str, dict[str, object]]:
        return {"buildings": {}}


def _controller() -> CentralizedMAPPOController:
    return CentralizedMAPPOController(
        _FakeEnv(),
        hidden_dimension=[8],
        critic_hidden_dimension=[16],
        rollout_steps=2,
        minibatch_size=2,
        n_epochs=1,
        normalize_rewards=False,
    )


def test_mappo_controller_updates_with_centralized_critic_observations() -> None:
    controller = _controller()
    observations = [
        [1.0, 2.0, 0.5, 0.0],
        [2.0, 3.0, 0.2, 0.0],
        [3.0, -1.0, 0.3, 0.0],
    ]

    for _ in range(2):
        payload = controller.sample_rollout_step(observations)
        assert payload["encoded_observations"].shape == (3, 8)
        assert payload["critic_observations"].shape == (3, 28)
        controller.store_rollout_step(
            step_payload=payload,
            rewards=[1.0, 0.5, -0.25],
            done=False,
        )

    stats = controller.finish_rollout(observations)

    assert stats["n_updates"] > 0
    assert controller.rollout_buffer.size == 0


def test_mappo_checkpoint_state_round_trips_critic_normalization() -> None:
    controller = _controller()
    observations = [
        [1.0, 2.0, 0.5, 0.0],
        [2.0, 3.0, 0.2, 0.0],
        [3.0, -1.0, 0.3, 0.0],
    ]
    controller.sample_rollout_step(observations)

    state = controller.checkpoint_state()
    restored = _controller()
    restored.load_checkpoint_state(state)

    assert restored.critic_obs_rms_mean is not None
    np.testing.assert_allclose(restored.critic_obs_rms_mean, controller.critic_obs_rms_mean)

from __future__ import annotations

import numpy as np

from cos435_citylearn.algorithms.mappo.rollout_buffer import CentralizedRolloutBuffer


def test_centralized_rollout_buffer_yields_actor_and_critic_batches() -> None:
    buffer = CentralizedRolloutBuffer(
        n_steps=2,
        n_buildings=3,
        actor_observation_dim=4,
        critic_observation_dim=9,
        action_dim=2,
    )
    rng = np.random.default_rng(0)
    for _ in range(2):
        buffer.add(
            actor_observations=rng.standard_normal((3, 4)).astype(np.float32),
            critic_observations=rng.standard_normal((3, 9)).astype(np.float32),
            pre_tanh_actions=rng.standard_normal((3, 2)).astype(np.float32),
            actions=rng.standard_normal((3, 2)).astype(np.float32),
            log_probs=rng.standard_normal(3).astype(np.float32),
            values=rng.standard_normal(3).astype(np.float32),
            rewards=rng.standard_normal(3).astype(np.float32),
            dones=np.zeros(3, dtype=np.float32),
        )
    buffer.compute_gae(np.zeros(3, dtype=np.float32), gamma=0.99, gae_lambda=0.95)

    seen = 0
    for batch in buffer.iter_minibatches(batch_size=2, shuffle=False, normalize_advantage=False):
        seen += batch["actor_observations"].shape[0]
        assert batch["actor_observations"].shape[1] == 4
        assert batch["critic_observations"].shape[1] == 9
        assert batch["pre_tanh_actions"].shape[1] == 2

    assert seen == 2 * 3


def test_centralized_rollout_buffer_gae_respects_done_boundary() -> None:
    buffer = CentralizedRolloutBuffer(
        n_steps=3,
        n_buildings=1,
        actor_observation_dim=1,
        critic_observation_dim=2,
        action_dim=1,
    )
    values = [1.0, 1.0, 1.0]
    rewards = [0.0, 10.0, 0.0]
    dones = [0.0, 1.0, 0.0]
    for t in range(3):
        buffer.add(
            actor_observations=np.zeros((1, 1), dtype=np.float32),
            critic_observations=np.zeros((1, 2), dtype=np.float32),
            pre_tanh_actions=np.zeros((1, 1), dtype=np.float32),
            actions=np.zeros((1, 1), dtype=np.float32),
            log_probs=np.zeros(1, dtype=np.float32),
            values=np.array([values[t]], dtype=np.float32),
            rewards=np.array([rewards[t]], dtype=np.float32),
            dones=np.array([dones[t]], dtype=np.float32),
        )

    buffer.compute_gae(np.array([1.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)

    assert np.isclose(buffer.advantages[1, 0], 9.0, atol=1e-5)

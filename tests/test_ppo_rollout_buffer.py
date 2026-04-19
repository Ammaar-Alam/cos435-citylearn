from __future__ import annotations

import numpy as np

from cos435_citylearn.algorithms.ppo.rollout_buffer import RolloutBuffer


def test_gae_matches_hand_calculation_single_building() -> None:
    buffer = RolloutBuffer(n_steps=4, n_buildings=1, observation_dim=2, action_dim=1)
    rewards = [1.0, 1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5, 0.5]
    for t in range(4):
        buffer.add(
            observations=np.zeros((1, 2), dtype=np.float32),
            pre_tanh_actions=np.zeros((1, 1), dtype=np.float32),
            actions=np.zeros((1, 1), dtype=np.float32),
            log_probs=np.zeros(1, dtype=np.float32),
            values=np.array([values[t]], dtype=np.float32),
            rewards=np.array([rewards[t]], dtype=np.float32),
            dones=np.array([0.0], dtype=np.float32),
        )
    last_value = np.array([0.5], dtype=np.float32)
    gamma = 1.0
    gae_lambda = 1.0

    buffer.compute_gae(last_value, gamma=gamma, gae_lambda=gae_lambda)

    # gamma=lambda=1, constant r=1, V=0.5 -> advantages count down 4,3,2,1
    expected_advantages = np.array([[4.0], [3.0], [2.0], [1.0]], dtype=np.float32)
    assert np.allclose(buffer.advantages, expected_advantages, atol=1e-5)
    expected_returns = expected_advantages + np.array([[0.5]] * 4, dtype=np.float32)
    assert np.allclose(buffer.returns, expected_returns, atol=1e-5)


def test_gae_respects_done_as_boundary() -> None:
    buffer = RolloutBuffer(n_steps=3, n_buildings=1, observation_dim=1, action_dim=1)
    values = [1.0, 1.0, 1.0]
    rewards = [0.0, 10.0, 0.0]
    dones = [0.0, 1.0, 0.0]  # done at t=1 so the bootstrap zeros out
    for t in range(3):
        buffer.add(
            observations=np.zeros((1, 1), dtype=np.float32),
            pre_tanh_actions=np.zeros((1, 1), dtype=np.float32),
            actions=np.zeros((1, 1), dtype=np.float32),
            log_probs=np.zeros(1, dtype=np.float32),
            values=np.array([values[t]], dtype=np.float32),
            rewards=np.array([rewards[t]], dtype=np.float32),
            dones=np.array([dones[t]], dtype=np.float32),
        )
    buffer.compute_gae(np.array([1.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)

    # done zeros the bootstrap so advantage = r - V = 10 - 1 = 9
    assert np.isclose(buffer.advantages[1, 0], 9.0, atol=1e-5)


def test_iter_minibatches_yields_flat_batches() -> None:
    buffer = RolloutBuffer(n_steps=2, n_buildings=3, observation_dim=4, action_dim=2)
    rng = np.random.default_rng(0)
    for _ in range(2):
        buffer.add(
            observations=rng.standard_normal((3, 4)).astype(np.float32),
            pre_tanh_actions=rng.standard_normal((3, 2)).astype(np.float32),
            actions=rng.standard_normal((3, 2)).astype(np.float32),
            log_probs=rng.standard_normal(3).astype(np.float32),
            values=rng.standard_normal(3).astype(np.float32),
            rewards=rng.standard_normal(3).astype(np.float32),
            dones=np.zeros(3, dtype=np.float32),
        )
    buffer.compute_gae(np.zeros(3, dtype=np.float32), gamma=0.99, gae_lambda=0.95)
    seen = 0
    for batch in buffer.iter_minibatches(batch_size=2, shuffle=False):
        seen += batch["observations"].shape[0]
        assert batch["observations"].shape[1] == 4
        assert batch["actions"].shape[1] == 2
    assert seen == 2 * 3

from __future__ import annotations

import torch

from cos435_citylearn.algorithms.ppo.networks import ActorNetwork


def test_sample_log_prob_matches_evaluate_actions() -> None:
    torch.manual_seed(0)
    observation_dim = 8
    action_dim = 3
    action_low = torch.tensor([-1.0, -0.5, 0.0])
    action_high = torch.tensor([1.0, 0.5, 1.0])
    actor = ActorNetwork(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=[32, 32],
        action_low=action_low,
        action_high=action_high,
    )
    observations = torch.randn(16, observation_dim)
    _, pre_tanh, log_prob_sample = actor.sample(observations)
    log_prob_eval, _ = actor.evaluate_actions(observations, pre_tanh)
    assert torch.allclose(log_prob_sample, log_prob_eval, atol=1e-5)


def test_near_bound_samples_produce_finite_log_probs() -> None:
    torch.manual_seed(1)
    observation_dim = 4
    action_dim = 2
    action_low = torch.tensor([-1.0, -1.0])
    action_high = torch.tensor([1.0, 1.0])
    actor = ActorNetwork(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=[16],
        action_low=action_low,
        action_high=action_high,
    )
    observations = torch.randn(8, observation_dim)
    extreme_pre_tanh = torch.full((8, action_dim), 6.0)
    log_prob, entropy = actor.evaluate_actions(observations, extreme_pre_tanh)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_deterministic_action_inside_bounds() -> None:
    torch.manual_seed(2)
    action_low = torch.tensor([-0.5, 0.0])
    action_high = torch.tensor([1.5, 2.0])
    actor = ActorNetwork(
        observation_dim=6,
        action_dim=2,
        hidden_dims=[8],
        action_low=action_low,
        action_high=action_high,
    )
    observations = torch.randn(32, 6)
    actions = actor.deterministic_action(observations)
    assert torch.all(actions >= action_low - 1e-5)
    assert torch.all(actions <= action_high + 1e-5)

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


def _mlp_layers(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.Tanh())
        prev = hidden
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        self.trunk = _mlp_layers(observation_dim, hidden_dims)
        trunk_output = hidden_dims[-1] if hidden_dims else observation_dim
        self.mean_head = nn.Linear(trunk_output, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.register_buffer("action_low", action_low.clone().detach())
        self.register_buffer("action_high", action_high.clone().detach())

    def _distribution(self, observation: torch.Tensor) -> Normal:
        features = self.trunk(observation)
        mean = self.mean_head(features)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def _squash(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        tanh_value = torch.tanh(pre_tanh)
        half_range = (self.action_high - self.action_low) * 0.5
        center = (self.action_high + self.action_low) * 0.5
        return tanh_value * half_range + center

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observation)
        pre_tanh = distribution.rsample()
        action = self._squash(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(distribution, pre_tanh)
        return action, pre_tanh, log_prob

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        distribution = self._distribution(observation)
        return self._squash(distribution.mean)

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        pre_tanh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observation)
        log_prob = self._log_prob_from_pre_tanh(distribution, pre_tanh)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def _log_prob_from_pre_tanh(
        self,
        distribution: Normal,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        base_log_prob = distribution.log_prob(pre_tanh).sum(dim=-1)
        tanh_correction = torch.log1p(-torch.tanh(pre_tanh).pow(2) + TANH_EPS).sum(dim=-1)
        half_range = (self.action_high - self.action_low) * 0.5
        affine_correction = torch.log(half_range + TANH_EPS).sum(dim=-1)
        return base_log_prob - tanh_correction - affine_correction


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.trunk = _mlp_layers(observation_dim, hidden_dims)
        trunk_output = hidden_dims[-1] if hidden_dims else observation_dim
        self.value_head = nn.Linear(trunk_output, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.trunk(observation)
        return self.value_head(features).squeeze(-1)

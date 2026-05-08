from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dimension: list[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dimension:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        last_dim = int(hidden_dim)
    return nn.Sequential(*layers), last_dim


class DeterministicActor(nn.Module):
    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_dimension: list[int],
        action_scaling_coefficient: float,
    ) -> None:
        super().__init__()
        self.trunk, last_dim = _mlp(observation_dimension, hidden_dimension)
        self.head = nn.Linear(last_dim, action_dimension)
        self.action_scaling_coefficient = float(action_scaling_coefficient)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_scaling_coefficient * torch.tanh(self.head(self.trunk(observation)))


class Critic(nn.Module):
    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_dimension: list[int],
    ) -> None:
        super().__init__()
        self.trunk, last_dim = _mlp(observation_dimension + action_dimension, hidden_dimension)
        self.head = nn.Linear(last_dim, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat([observation, action], dim=1)))

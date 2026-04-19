from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    n_steps: int
    n_buildings: int
    observation_dim: int
    action_dim: int

    observations: np.ndarray = field(init=False)
    pre_tanh_actions: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)
    _index: int = field(init=False)

    def __post_init__(self) -> None:
        shape_obs = (self.n_steps, self.n_buildings, self.observation_dim)
        shape_act = (self.n_steps, self.n_buildings, self.action_dim)
        shape_scalar = (self.n_steps, self.n_buildings)

        self.observations = np.zeros(shape_obs, dtype=np.float32)
        self.pre_tanh_actions = np.zeros(shape_act, dtype=np.float32)
        self.actions = np.zeros(shape_act, dtype=np.float32)
        self.log_probs = np.zeros(shape_scalar, dtype=np.float32)
        self.values = np.zeros(shape_scalar, dtype=np.float32)
        self.rewards = np.zeros(shape_scalar, dtype=np.float32)
        self.dones = np.zeros(shape_scalar, dtype=np.float32)
        self.advantages = np.zeros(shape_scalar, dtype=np.float32)
        self.returns = np.zeros(shape_scalar, dtype=np.float32)
        self._index = 0

    @property
    def size(self) -> int:
        return self._index

    @property
    def full(self) -> bool:
        return self._index >= self.n_steps

    def reset(self) -> None:
        self._index = 0

    def add(
        self,
        *,
        observations: np.ndarray,
        pre_tanh_actions: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if self._index >= self.n_steps:
            raise RuntimeError("rollout buffer is already full")

        step = self._index
        self.observations[step] = observations
        self.pre_tanh_actions[step] = pre_tanh_actions
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.values[step] = values
        self.rewards[step] = rewards
        self.dones[step] = dones
        self._index += 1

    def compute_gae(
        self,
        last_values: np.ndarray,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        n = self.size
        if n == 0:
            return

        advantages = np.zeros((n, self.n_buildings), dtype=np.float32)
        last_gae = np.zeros(self.n_buildings, dtype=np.float32)

        for step in reversed(range(n)):
            # done zeros the bootstrap so the next episode doesn't leak in
            next_non_terminal = 1.0 - self.dones[step]
            if step == n - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae

        self.advantages[:n] = advantages
        self.returns[:n] = advantages + self.values[:n]

    def iter_minibatches(
        self,
        *,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device | None = None,
    ):
        n = self.size * self.n_buildings
        if n == 0:
            return

        flat_observations = self.observations[: self.size].reshape(n, self.observation_dim)
        flat_pre_tanh = self.pre_tanh_actions[: self.size].reshape(n, self.action_dim)
        flat_actions = self.actions[: self.size].reshape(n, self.action_dim)
        flat_log_probs = self.log_probs[: self.size].reshape(n)
        flat_values = self.values[: self.size].reshape(n)
        flat_advantages = self.advantages[: self.size].reshape(n)
        flat_returns = self.returns[: self.size].reshape(n)

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        batch_size = max(1, min(batch_size, n))
        for start in range(0, n, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield {
                "observations": torch.as_tensor(flat_observations[batch_indices], device=device),
                "pre_tanh_actions": torch.as_tensor(flat_pre_tanh[batch_indices], device=device),
                "actions": torch.as_tensor(flat_actions[batch_indices], device=device),
                "log_probs": torch.as_tensor(flat_log_probs[batch_indices], device=device),
                "values": torch.as_tensor(flat_values[batch_indices], device=device),
                "advantages": torch.as_tensor(flat_advantages[batch_indices], device=device),
                "returns": torch.as_tensor(flat_returns[batch_indices], device=device),
            }

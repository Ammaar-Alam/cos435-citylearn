from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from citylearn.agents.rlc import RLC
from citylearn.preprocessing import Encoder, RemoveFeature
from citylearn.rl import ReplayBuffer
from torch import nn, optim

from cos435_citylearn.algorithms.ppo.shared_features import (
    SHARED_CONTEXT_V2_DIMENSION,
    build_shared_context_v2,
)
from cos435_citylearn.algorithms.td3.networks import Critic, DeterministicActor


def _array_to_list(values: Any) -> Any:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        return values.tolist()
    return values


def _tensor(value: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def _restore_time_state(controller: Any, *, restored_time_step: int) -> None:
    controller._Environment__time_step = restored_time_step
    controller._Agent__actions = [
        [[] for _ in range(restored_time_step + 1)] for _ in controller.action_space
    ]


class SharedTD3Controller(RLC):
    def __init__(
        self,
        env,
        *,
        hidden_dimension: list[int] | None = None,
        discount: float | None = None,
        tau: float | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        replay_buffer_capacity: int | None = None,
        standardize_start_time_step: int | None = None,
        end_exploration_time_step: int | None = None,
        action_scaling_coefficienct: float | None = None,
        reward_scaling: float | None = None,
        update_per_time_step: int | None = None,
        shared_context_dimension: int = SHARED_CONTEXT_V2_DIMENSION,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
    ) -> None:
        self.shared_context_dimension = int(shared_context_dimension)
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.exploration_noise = float(exploration_noise)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic_criterion = nn.MSELoss()
        self.normalized = False
        self.total_updates = 0
        self.last_update_stats: dict[str, float] = {}
        if self.shared_context_dimension != SHARED_CONTEXT_V2_DIMENSION:
            raise ValueError(
                f"shared_context_dimension must be {SHARED_CONTEXT_V2_DIMENSION} "
                "for shared TD3 context v2"
            )

        super().__init__(
            env,
            hidden_dimension=hidden_dimension,
            discount=discount,
            tau=tau,
            lr=lr,
            batch_size=batch_size,
            replay_buffer_capacity=replay_buffer_capacity,
            standardize_start_time_step=standardize_start_time_step,
            end_exploration_time_step=end_exploration_time_step,
            action_scaling_coefficienct=action_scaling_coefficienct,
            reward_scaling=reward_scaling,
            update_per_time_step=update_per_time_step,
        )
        self.replay_buffer = ReplayBuffer(int(self.replay_buffer_capacity))
        self.norm_mean: np.ndarray | None = None
        self.norm_std: np.ndarray | None = None
        self.r_norm_mean: float | None = None
        self.r_norm_std: float | None = None
        self.actor: DeterministicActor | None = None
        self.critic1: Critic | None = None
        self.critic2: Critic | None = None
        self.target_actor: DeterministicActor | None = None
        self.target_critic1: Critic | None = None
        self.target_critic2: Critic | None = None
        self.actor_optimizer: optim.Optimizer | None = None
        self.critic1_optimizer: optim.Optimizer | None = None
        self.critic2_optimizer: optim.Optimizer | None = None
        self._validate_action_spaces()
        self._set_networks()

    @property
    def controller_type(self) -> str:
        return "shared_parameter_td3"

    def _validate_action_spaces(self) -> None:
        reference_dimension = self.action_dimension[0]
        reference_shape = tuple(self.action_space[0].shape)
        for action_dimension, space in zip(self.action_dimension, self.action_space):
            if action_dimension != reference_dimension or tuple(space.shape) != reference_shape:
                raise ValueError("shared TD3 requires identical per-building action spaces")

    def set_encoders(self) -> list[list[Encoder]]:
        encoders = super().set_encoders()
        for index, names in enumerate(self.observation_names):
            for feature_index, name in enumerate(names):
                if name == "net_electricity_consumption":
                    encoders[index][feature_index] = RemoveFeature()
        return encoders

    def _observation_dimension(self) -> int:
        encoded_lengths = []
        for encoders, space in zip(self.encoders, self.observation_space):
            encoded = [
                value
                for value in np.hstack(encoders * np.ones(len(space.low)))
                if value is not None
            ]
            encoded_lengths.append(len(encoded))
        if len(set(encoded_lengths)) != 1:
            raise ValueError("shared TD3 requires identical per-building observation sizes")
        return int(encoded_lengths[0]) + self.shared_context_dimension

    def _set_networks(self) -> None:
        observation_dimension = self._observation_dimension()
        action_dimension = self.action_dimension[0]
        self.actor = DeterministicActor(
            observation_dimension,
            action_dimension,
            self.hidden_dimension,
            self.action_scaling_coefficient,
        ).to(self.device)
        self.target_actor = DeterministicActor(
            observation_dimension,
            action_dimension,
            self.hidden_dimension,
            self.action_scaling_coefficient,
        ).to(self.device)
        self.critic1 = Critic(observation_dimension, action_dimension, self.hidden_dimension).to(
            self.device
        )
        self.critic2 = Critic(observation_dimension, action_dimension, self.hidden_dimension).to(
            self.device
        )
        self.target_critic1 = Critic(
            observation_dimension, action_dimension, self.hidden_dimension
        ).to(self.device)
        self.target_critic2 = Critic(
            observation_dimension, action_dimension, self.hidden_dimension
        ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)

    def _encode_observation(
        self, index: int, observation: Sequence[float], shared_context: np.ndarray
    ) -> np.ndarray:
        encoded = np.asarray(
            [
                value
                for value in np.hstack(
                    self.encoders[index] * np.asarray(observation, dtype="float32")
                )
                if value is not None
            ],
            dtype="float32",
        )
        return np.concatenate([encoded, shared_context], dtype="float32")

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.norm_mean is None or self.norm_std is None:
            return observation
        return (observation - self.norm_mean) / self.norm_std

    def _normalize_reward(self, reward: float) -> float:
        if self.r_norm_mean is None or self.r_norm_std is None:
            return float(reward)
        return float((reward - self.r_norm_mean) / self.r_norm_std)

    def predict(
        self, observations: list[list[float]], deterministic: bool = False
    ) -> list[list[float]]:
        if self.time_step <= self.end_exploration_time_step and not deterministic:
            actions = [
                list(self.action_scaling_coefficient * space.sample())
                for space in self.action_space
            ]
        else:
            actions = self._predict_with_actor(observations, deterministic=deterministic)
        self.actions = actions
        self.next_time_step()
        return actions

    def _predict_with_actor(
        self,
        observations: list[list[float]],
        *,
        deterministic: bool,
    ) -> list[list[float]]:
        shared_context = build_shared_context_v2(observations, self.observation_names)
        actions: list[list[float]] = []
        for index, observation in enumerate(observations):
            encoded = self._encode_observation(index, observation, shared_context)
            encoded = self._normalize_observation(encoded)
            observation_tensor = _tensor(encoded, device=self.device).unsqueeze(0)
            action = self.actor(observation_tensor).detach().cpu().numpy()[0]
            if not deterministic and self.exploration_noise > 0:
                action = action + np.random.normal(
                    loc=0.0,
                    scale=self.exploration_noise,
                    size=action.shape,
                )
            action = np.clip(action, self.action_space[index].low, self.action_space[index].high)
            actions.append(action.astype(float).tolist())
        return actions

    def update(
        self,
        observations: list[list[float]],
        actions: list[list[float]],
        reward: list[float],
        next_observations: list[list[float]],
        done: bool,
    ) -> None:
        shared_context = build_shared_context_v2(observations, self.observation_names)
        next_shared_context = build_shared_context_v2(next_observations, self.observation_names)

        for index, (observation, action, reward_value, next_observation) in enumerate(
            zip(observations, actions, reward, next_observations)
        ):
            encoded_observation = self._encode_observation(index, observation, shared_context)
            encoded_next_observation = self._encode_observation(
                index, next_observation, next_shared_context
            )
            if self.normalized:
                encoded_observation = self._normalize_observation(encoded_observation)
                encoded_next_observation = self._normalize_observation(encoded_next_observation)
                reward_value = self._normalize_reward(reward_value)

            self.replay_buffer.push(
                encoded_observation,
                np.asarray(action, dtype="float32"),
                float(reward_value),
                encoded_next_observation,
                done,
            )

        if self.time_step < self.standardize_start_time_step:
            return
        if self.batch_size > len(self.replay_buffer):
            return

        if not self.normalized:
            observations_batch = np.asarray(
                [item[0] for item in self.replay_buffer.buffer],
                dtype="float32",
            )
            self.norm_mean = np.nanmean(observations_batch, axis=0)
            self.norm_std = np.nanstd(observations_batch, axis=0) + 1e-5
            rewards_batch = np.asarray(
                [item[2] for item in self.replay_buffer.buffer], dtype="float32"
            )
            self.r_norm_mean = float(np.nanmean(rewards_batch, dtype="float32"))
            self.r_norm_std = (
                float(np.nanstd(rewards_batch, dtype="float32")) / self.reward_scaling + 1e-5
            )
            self.replay_buffer.buffer = [
                (
                    self._normalize_observation(sample_observation),
                    sample_action,
                    self._normalize_reward(sample_reward),
                    self._normalize_observation(sample_next_observation),
                    sample_done,
                )
                for (
                    sample_observation,
                    sample_action,
                    sample_reward,
                    sample_next_observation,
                    sample_done,
                ) in self.replay_buffer.buffer
            ]
            self.normalized = True

        stats: list[dict[str, float]] = []
        for _ in range(self.update_per_time_step):
            (
                sample_observation,
                sample_action,
                sample_reward,
                sample_next_observation,
                sample_done,
            ) = self.replay_buffer.sample(self.batch_size)
            observation_tensor = _tensor(sample_observation, device=self.device)
            next_observation_tensor = _tensor(sample_next_observation, device=self.device)
            action_tensor = _tensor(sample_action, device=self.device)
            reward_tensor = _tensor(sample_reward, device=self.device).unsqueeze(1)
            done_tensor = _tensor(sample_done, device=self.device).unsqueeze(1)

            with torch.no_grad():
                target_action = self.target_actor(next_observation_tensor)
                if self.target_policy_noise > 0:
                    noise = torch.randn_like(target_action) * self.target_policy_noise
                    noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
                    target_action = target_action + noise
                low = torch.as_tensor(
                    self.action_space[0].low,
                    dtype=torch.float32,
                    device=self.device,
                )
                high = torch.as_tensor(
                    self.action_space[0].high,
                    dtype=torch.float32,
                    device=self.device,
                )
                target_action = torch.max(torch.min(target_action, high), low)
                target_q = torch.min(
                    self.target_critic1(next_observation_tensor, target_action),
                    self.target_critic2(next_observation_tensor, target_action),
                )
                q_target = reward_tensor + (1.0 - done_tensor) * self.discount * target_q

            q1_prediction = self.critic1(observation_tensor, action_tensor)
            q2_prediction = self.critic2(observation_tensor, action_tensor)
            q1_loss = self.critic_criterion(q1_prediction, q_target)
            q2_loss = self.critic_criterion(q2_prediction, q_target)
            self.critic1_optimizer.zero_grad()
            q1_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.zero_grad()
            q2_loss.backward()
            self.critic2_optimizer.step()

            policy_loss_value = 0.0
            self.total_updates += 1
            if self.total_updates % self.policy_delay == 0:
                actor_action = self.actor(observation_tensor)
                policy_loss = -self.critic1(observation_tensor, actor_action).mean()
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()
                policy_loss_value = float(policy_loss.detach().cpu().item())
                _soft_update(self.target_actor, self.actor, self.tau)
                _soft_update(self.target_critic1, self.critic1, self.tau)
                _soft_update(self.target_critic2, self.critic2, self.tau)

            stats.append(
                {
                    "q1_loss": float(q1_loss.detach().cpu().item()),
                    "q2_loss": float(q2_loss.detach().cpu().item()),
                    "policy_loss": policy_loss_value,
                    "alpha": 0.0,
                    "alpha_loss": 0.0,
                    "buffer_size": float(len(self.replay_buffer)),
                }
            )

        self.last_update_stats = {
            key: float(np.mean([item[key] for item in stats], dtype="float32")) for key in stats[0]
        }

    def training_stats(self) -> dict[str, float]:
        return dict(self.last_update_stats)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "controller_type": self.controller_type,
            "hidden_dimension": list(self.hidden_dimension),
            "discount": float(self.discount),
            "tau": float(self.tau),
            "lr": float(self.lr),
            "batch_size": int(self.batch_size),
            "replay_buffer_capacity": int(self.replay_buffer_capacity),
            "standardize_start_time_step": int(self.standardize_start_time_step),
            "end_exploration_time_step": int(self.end_exploration_time_step),
            "action_scaling_coefficient": float(self.action_scaling_coefficient),
            "reward_scaling": float(self.reward_scaling),
            "update_per_time_step": int(self.update_per_time_step),
            "policy_delay": int(self.policy_delay),
            "target_policy_noise": float(self.target_policy_noise),
            "target_noise_clip": float(self.target_noise_clip),
            "exploration_noise": float(self.exploration_noise),
            "shared_context_dimension": int(self.shared_context_dimension),
            "shared_context_version": "v2",
            "time_step": int(self.time_step),
            "total_updates": int(self.total_updates),
            "normalized": bool(self.normalized),
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "norm_mean": _array_to_list(self.norm_mean),
            "norm_std": _array_to_list(self.norm_std),
            "r_norm_mean": self.r_norm_mean,
            "r_norm_std": self.r_norm_std,
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        _restore_time_state(self, restored_time_step=int(payload["time_step"]))
        self.normalized = bool(payload["normalized"])
        self.total_updates = int(payload["total_updates"])
        self.norm_mean = (
            None
            if payload["norm_mean"] is None
            else np.asarray(payload["norm_mean"], dtype="float32")
        )
        self.norm_std = (
            None
            if payload["norm_std"] is None
            else np.asarray(payload["norm_std"], dtype="float32")
        )
        self.r_norm_mean = None if payload["r_norm_mean"] is None else float(payload["r_norm_mean"])
        self.r_norm_std = None if payload["r_norm_std"] is None else float(payload["r_norm_std"])
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic1.load_state_dict(payload["critic1_state_dict"])
        self.critic2.load_state_dict(payload["critic2_state_dict"])
        self.target_actor.load_state_dict(payload["target_actor_state_dict"])
        self.target_critic1.load_state_dict(payload["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(payload["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(payload["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(payload["critic2_optimizer_state_dict"])

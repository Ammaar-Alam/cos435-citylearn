from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from citylearn.agents.rlc import RLC
from citylearn.agents.sac import SAC as CityLearnSAC
from citylearn.preprocessing import Encoder, RemoveFeature
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork
from torch import nn, optim

from cos435_citylearn.algorithms.sac.features import (
    SHARED_CONTEXT_DIMENSION,
    build_shared_context,
)


def _array_to_list(values: Any) -> Any:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        return values.tolist()
    return values


def _tensor(value: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _require_checkpoint_list_length(
    field_name: str,
    values: Sequence[Any],
    *,
    expected_count: int,
) -> None:
    actual_count = len(values)
    if actual_count != expected_count:
        raise ValueError(
            f"SAC checkpoint {field_name} count {actual_count} does not match expected controller count {expected_count}"
        )


class CentralizedSACController(CityLearnSAC):
    def __init__(self, env, *, auto_entropy_tuning: bool = True, **kwargs: Any):
        self.auto_entropy_tuning = auto_entropy_tuning
        self.last_update_stats: dict[str, float] = {}
        # This keeps CityLearn's native SAC data path and network definitions, but
        # extends the update step with entropy tuning, stats capture, and checkpoint I/O.
        super().__init__(env, **kwargs)
        self.log_alpha: list[torch.Tensor | None] = [None for _ in self.action_space]
        self.alpha_optimizer: list[optim.Optimizer | None] = [None for _ in self.action_space]

        if self.auto_entropy_tuning:
            for index in range(len(self.action_space)):
                initial_alpha = float(self.alpha)
                log_alpha = torch.tensor(
                    np.log(max(initial_alpha, 1e-6)),
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
                self.log_alpha[index] = log_alpha
                self.alpha_optimizer[index] = optim.Adam([log_alpha], lr=self.lr)

    @property
    def controller_type(self) -> str:
        return "centralized_native"

    def _alpha_value(self, index: int) -> torch.Tensor | float:
        if not self.auto_entropy_tuning or self.log_alpha[index] is None:
            return float(self.alpha)
        return self.log_alpha[index].exp()

    def update(
        self,
        observations: list[list[float]],
        actions: list[list[float]],
        reward: list[float],
        next_observations: list[list[float]],
        done: bool,
    ) -> None:
        stats: list[dict[str, float]] = []

        for index, (observation, action, reward_value, next_observation) in enumerate(
            zip(observations, actions, reward, next_observations)
        ):
            encoded_observation = self.get_encoded_observations(index, observation)
            encoded_next_observation = self.get_encoded_observations(index, next_observation)

            if self.normalized[index]:
                encoded_observation = self.get_normalized_observations(index, encoded_observation)
                encoded_next_observation = self.get_normalized_observations(
                    index, encoded_next_observation
                )
                reward_value = self.get_normalized_reward(index, reward_value)

            self.replay_buffer[index].push(
                encoded_observation,
                action,
                reward_value,
                encoded_next_observation,
                done,
            )

            if self.time_step < self.standardize_start_time_step:
                continue
            if self.batch_size > len(self.replay_buffer[index]):
                continue

            if not self.normalized[index]:
                observations_batch = np.asarray(
                    [item[0] for item in self.replay_buffer[index].buffer],
                    dtype="float32",
                )
                self.norm_mean[index] = np.nanmean(observations_batch, axis=0)
                self.norm_std[index] = np.nanstd(observations_batch, axis=0) + 1e-5
                rewards_batch = np.asarray(
                    [item[2] for item in self.replay_buffer[index].buffer],
                    dtype="float32",
                )
                self.r_norm_mean[index] = float(np.nanmean(rewards_batch, dtype="float32"))
                self.r_norm_std[index] = (
                    float(np.nanstd(rewards_batch, dtype="float32")) / self.reward_scaling + 1e-5
                )
                self.replay_buffer[index].buffer = [
                    (
                        np.hstack(self.get_normalized_observations(index, sample_observation).reshape(1, -1)[0]),
                        sample_action,
                        self.get_normalized_reward(index, sample_reward),
                        np.hstack(
                            self.get_normalized_observations(index, sample_next_observation).reshape(1, -1)[0]
                        ),
                        sample_done,
                    )
                    for sample_observation, sample_action, sample_reward, sample_next_observation, sample_done in self.replay_buffer[index].buffer
                ]
                self.normalized[index] = True

            for _ in range(self.update_per_time_step):
                sample_observation, sample_action, sample_reward, sample_next_observation, sample_done = self.replay_buffer[
                    index
                ].sample(self.batch_size)
                observation_tensor = _tensor(sample_observation, device=self.device)
                next_observation_tensor = _tensor(sample_next_observation, device=self.device)
                action_tensor = _tensor(sample_action, device=self.device)
                reward_tensor = _tensor(sample_reward, device=self.device).unsqueeze(1)
                done_tensor = _tensor(sample_done, device=self.device).unsqueeze(1)

                with torch.no_grad():
                    next_action, next_log_pi, _ = self.policy_net[index].sample(next_observation_tensor)
                    alpha_value = self._alpha_value(index)
                    target_q = torch.min(
                        self.target_soft_q_net1[index](next_observation_tensor, next_action),
                        self.target_soft_q_net2[index](next_observation_tensor, next_action),
                    ) - alpha_value * next_log_pi
                    q_target = reward_tensor + (1.0 - done_tensor) * self.discount * target_q

                q1_prediction = self.soft_q_net1[index](observation_tensor, action_tensor)
                q2_prediction = self.soft_q_net2[index](observation_tensor, action_tensor)
                q1_loss = self.soft_q_criterion(q1_prediction, q_target)
                q2_loss = self.soft_q_criterion(q2_prediction, q_target)
                self.soft_q_optimizer1[index].zero_grad()
                q1_loss.backward()
                self.soft_q_optimizer1[index].step()
                self.soft_q_optimizer2[index].zero_grad()
                q2_loss.backward()
                self.soft_q_optimizer2[index].step()

                new_action, log_pi, _ = self.policy_net[index].sample(observation_tensor)
                q_new_action = torch.min(
                    self.soft_q_net1[index](observation_tensor, new_action),
                    self.soft_q_net2[index](observation_tensor, new_action),
                )
                alpha_value = self._alpha_value(index)
                policy_loss = (alpha_value * log_pi - q_new_action).mean()
                self.policy_optimizer[index].zero_grad()
                policy_loss.backward()
                self.policy_optimizer[index].step()

                alpha_loss_value = 0.0
                if self.auto_entropy_tuning and self.log_alpha[index] is not None:
                    alpha_loss = -(
                        self.log_alpha[index] * (log_pi + self.target_entropy[index]).detach()
                    ).mean()
                    self.alpha_optimizer[index].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer[index].step()
                    alpha_loss_value = float(alpha_loss.detach().cpu().item())
                    alpha_value = self._alpha_value(index)

                for target_param, param in zip(
                    self.target_soft_q_net1[index].parameters(),
                    self.soft_q_net1[index].parameters(),
                ):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                for target_param, param in zip(
                    self.target_soft_q_net2[index].parameters(),
                    self.soft_q_net2[index].parameters(),
                ):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                stats.append(
                    {
                        "q1_loss": float(q1_loss.detach().cpu().item()),
                        "q2_loss": float(q2_loss.detach().cpu().item()),
                        "policy_loss": float(policy_loss.detach().cpu().item()),
                        "alpha": float(alpha_value.detach().cpu().item())
                        if isinstance(alpha_value, torch.Tensor)
                        else float(alpha_value),
                        "alpha_loss": alpha_loss_value,
                        "buffer_size": float(len(self.replay_buffer[index])),
                    }
                )

        if stats:
            self.last_update_stats = {
                key: float(np.mean([item[key] for item in stats], dtype="float32"))
                for key in stats[0]
            }

    def training_stats(self) -> dict[str, float]:
        return dict(self.last_update_stats)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "controller_type": self.controller_type,
            "hidden_dimension": list(self.hidden_dimension),
            "discount": float(self.discount),
            "tau": float(self.tau),
            "alpha": float(self.alpha),
            "lr": float(self.lr),
            "batch_size": int(self.batch_size),
            "replay_buffer_capacity": int(self.replay_buffer_capacity),
            "standardize_start_time_step": int(self.standardize_start_time_step),
            "end_exploration_time_step": int(self.end_exploration_time_step),
            "action_scaling_coefficient": float(self.action_scaling_coefficient),
            "reward_scaling": float(self.reward_scaling),
            "update_per_time_step": int(self.update_per_time_step),
            "auto_entropy_tuning": bool(self.auto_entropy_tuning),
            "time_step": int(self.time_step),
            "normalized": list(self.normalized),
            "policy_state_dicts": [network.state_dict() for network in self.policy_net],
            "soft_q1_state_dicts": [network.state_dict() for network in self.soft_q_net1],
            "soft_q2_state_dicts": [network.state_dict() for network in self.soft_q_net2],
            "target_soft_q1_state_dicts": [network.state_dict() for network in self.target_soft_q_net1],
            "target_soft_q2_state_dicts": [network.state_dict() for network in self.target_soft_q_net2],
            "policy_optimizer_state_dicts": [optimizer.state_dict() for optimizer in self.policy_optimizer],
            "soft_q_optimizer1_state_dicts": [optimizer.state_dict() for optimizer in self.soft_q_optimizer1],
            "soft_q_optimizer2_state_dicts": [optimizer.state_dict() for optimizer in self.soft_q_optimizer2],
            "log_alpha": [
                None if value is None else float(value.detach().cpu().item())
                for value in self.log_alpha
            ],
            "alpha_optimizer_state_dicts": [
                None if optimizer is None else optimizer.state_dict()
                for optimizer in self.alpha_optimizer
            ],
            "norm_mean": [_array_to_list(value) for value in self.norm_mean],
            "norm_std": [_array_to_list(value) for value in self.norm_std],
            "r_norm_mean": [_array_to_list(value) for value in self.r_norm_mean],
            "r_norm_std": [_array_to_list(value) for value in self.r_norm_std],
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        expected_count = len(self.policy_net)
        counted_fields = (
            "normalized",
            "policy_state_dicts",
            "soft_q1_state_dicts",
            "soft_q2_state_dicts",
            "target_soft_q1_state_dicts",
            "target_soft_q2_state_dicts",
            "policy_optimizer_state_dicts",
            "soft_q_optimizer1_state_dicts",
            "soft_q_optimizer2_state_dicts",
            "norm_mean",
            "norm_std",
            "r_norm_mean",
            "r_norm_std",
        )

        for field_name in counted_fields:
            _require_checkpoint_list_length(
                field_name,
                payload[field_name],
                expected_count=expected_count,
            )

        if self.auto_entropy_tuning:
            _require_checkpoint_list_length(
                "log_alpha",
                payload["log_alpha"],
                expected_count=expected_count,
            )
            _require_checkpoint_list_length(
                "alpha_optimizer_state_dicts",
                payload["alpha_optimizer_state_dicts"],
                expected_count=expected_count,
            )

        self.normalized = list(payload["normalized"])
        self.norm_mean = [
            None if value is None else np.asarray(value, dtype="float32")
            for value in payload["norm_mean"]
        ]
        self.norm_std = [
            None if value is None else np.asarray(value, dtype="float32")
            for value in payload["norm_std"]
        ]
        self.r_norm_mean = [
            None if value is None else float(value)
            for value in payload["r_norm_mean"]
        ]
        self.r_norm_std = [
            None if value is None else float(value)
            for value in payload["r_norm_std"]
        ]

        for network, state_dict in zip(self.policy_net, payload["policy_state_dicts"]):
            network.load_state_dict(state_dict)
        for network, state_dict in zip(self.soft_q_net1, payload["soft_q1_state_dicts"]):
            network.load_state_dict(state_dict)
        for network, state_dict in zip(self.soft_q_net2, payload["soft_q2_state_dicts"]):
            network.load_state_dict(state_dict)
        for network, state_dict in zip(
            self.target_soft_q_net1, payload["target_soft_q1_state_dicts"]
        ):
            network.load_state_dict(state_dict)
        for network, state_dict in zip(
            self.target_soft_q_net2, payload["target_soft_q2_state_dicts"]
        ):
            network.load_state_dict(state_dict)
        for optimizer, state_dict in zip(
            self.policy_optimizer, payload["policy_optimizer_state_dicts"]
        ):
            optimizer.load_state_dict(state_dict)
        for optimizer, state_dict in zip(
            self.soft_q_optimizer1, payload["soft_q_optimizer1_state_dicts"]
        ):
            optimizer.load_state_dict(state_dict)
        for optimizer, state_dict in zip(
            self.soft_q_optimizer2, payload["soft_q_optimizer2_state_dicts"]
        ):
            optimizer.load_state_dict(state_dict)

        if self.auto_entropy_tuning:
            for index, value in enumerate(payload["log_alpha"]):
                if value is None:
                    continue
                self.log_alpha[index] = torch.tensor(
                    float(value),
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
                self.alpha_optimizer[index] = optim.Adam([self.log_alpha[index]], lr=self.lr)
                state_dict = payload["alpha_optimizer_state_dicts"][index]
                if state_dict is not None:
                    self.alpha_optimizer[index].load_state_dict(state_dict)


class SharedSACController(RLC):
    def __init__(
        self,
        env,
        *,
        hidden_dimension: list[int] | None = None,
        discount: float | None = None,
        tau: float | None = None,
        alpha: float | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        replay_buffer_capacity: int | None = None,
        standardize_start_time_step: int | None = None,
        end_exploration_time_step: int | None = None,
        action_scaling_coefficienct: float | None = None,
        reward_scaling: float | None = None,
        update_per_time_step: int | None = None,
        shared_context_dimension: int = 4,
        auto_entropy_tuning: bool = True,
    ):
        self.shared_context_dimension = int(shared_context_dimension)
        self.auto_entropy_tuning = auto_entropy_tuning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_q_criterion = nn.SmoothL1Loss()
        self.normalized = False
        self.last_update_stats: dict[str, float] = {}
        if self.shared_context_dimension != SHARED_CONTEXT_DIMENSION:
            raise ValueError(
                f"shared_context_dimension must be {SHARED_CONTEXT_DIMENSION} for the current shared SAC feature set"
            )
        # This follows the same encoder/replay/network conventions as CityLearn's
        # native SAC, but uses one shared actor-critic pair across all buildings.
        super().__init__(
            env,
            hidden_dimension=hidden_dimension,
            discount=discount,
            tau=tau,
            alpha=alpha,
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
        self.soft_q_net1: SoftQNetwork | None = None
        self.soft_q_net2: SoftQNetwork | None = None
        self.target_soft_q_net1: SoftQNetwork | None = None
        self.target_soft_q_net2: SoftQNetwork | None = None
        self.policy_net: PolicyNetwork | None = None
        self.soft_q_optimizer1: optim.Optimizer | None = None
        self.soft_q_optimizer2: optim.Optimizer | None = None
        self.policy_optimizer: optim.Optimizer | None = None
        self.log_alpha: torch.Tensor | None = None
        self.alpha_optimizer: optim.Optimizer | None = None
        self.target_entropy: float | None = None
        self._validate_action_spaces()
        self._set_networks()

    @property
    def controller_type(self) -> str:
        return "shared_parameter_sac"

    def _validate_action_spaces(self) -> None:
        reference_dimension = self.action_dimension[0]
        reference_shape = tuple(self.action_space[0].shape)

        for action_dimension, space in zip(self.action_dimension, self.action_space):
            if action_dimension != reference_dimension or tuple(space.shape) != reference_shape:
                raise ValueError("shared SAC requires identical per-building action spaces")

    def _observation_dimension(self) -> int:
        encoded_lengths = []

        for encoders, space in zip(self.encoders, self.observation_space):
            encoded = [value for value in np.hstack(encoders * np.ones(len(space.low))) if value is not None]
            encoded_lengths.append(len(encoded))

        if len(set(encoded_lengths)) != 1:
            raise ValueError("shared SAC requires identical per-building encoded observation sizes")

        return int(encoded_lengths[0]) + self.shared_context_dimension

    def _set_networks(self) -> None:
        observation_dimension = self._observation_dimension()
        action_dimension = self.action_dimension[0]
        action_space = self.action_space[0]
        self.soft_q_net1 = SoftQNetwork(observation_dimension, action_dimension, self.hidden_dimension).to(
            self.device
        )
        self.soft_q_net2 = SoftQNetwork(observation_dimension, action_dimension, self.hidden_dimension).to(
            self.device
        )
        self.target_soft_q_net1 = SoftQNetwork(
            observation_dimension, action_dimension, self.hidden_dimension
        ).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(
            observation_dimension, action_dimension, self.hidden_dimension
        ).to(self.device)
        self.target_soft_q_net1.load_state_dict(self.soft_q_net1.state_dict())
        self.target_soft_q_net2.load_state_dict(self.soft_q_net2.state_dict())
        self.policy_net = PolicyNetwork(
            observation_dimension,
            action_dimension,
            action_space,
            self.action_scaling_coefficient,
            self.hidden_dimension,
        ).to(self.device)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.target_entropy = -float(np.prod(action_space.shape).item())
        if self.auto_entropy_tuning:
            self.log_alpha = torch.tensor(
                np.log(max(float(self.alpha), 1e-6)),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

    def set_encoders(self) -> list[list[Encoder]]:
        encoders = super().set_encoders()

        for index, names in enumerate(self.observation_names):
            for feature_index, name in enumerate(names):
                if name == "net_electricity_consumption":
                    encoders[index][feature_index] = RemoveFeature()

        return encoders

    def _encode_observation(self, index: int, observation: Sequence[float], shared_context: np.ndarray) -> np.ndarray:
        encoded = np.asarray(
            [value for value in np.hstack(self.encoders[index] * np.asarray(observation, dtype="float32")) if value is not None],
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

    def _alpha_value(self) -> torch.Tensor | float:
        if not self.auto_entropy_tuning or self.log_alpha is None:
            return float(self.alpha)
        return self.log_alpha.exp()

    def predict(self, observations: list[list[float]], deterministic: bool = False) -> list[list[float]]:
        if self.time_step > self.end_exploration_time_step or deterministic:
            actions = self._predict_with_policy(observations, deterministic=deterministic)
        else:
            actions = [list(self.action_scaling_coefficient * space.sample()) for space in self.action_space]

        self.actions = actions
        self.next_time_step()
        return actions

    def _predict_with_policy(
        self,
        observations: list[list[float]],
        *,
        deterministic: bool,
    ) -> list[list[float]]:
        shared_context = build_shared_context(observations, self.observation_names)
        actions: list[list[float]] = []

        for index, observation in enumerate(observations):
            encoded = self._encode_observation(index, observation, shared_context)
            encoded = self._normalize_observation(encoded)
            observation_tensor = torch.as_tensor(
                encoded,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            sampled_action, _, mean_action = self.policy_net.sample(observation_tensor)
            action_tensor = mean_action if deterministic else sampled_action
            actions.append(action_tensor.detach().cpu().numpy()[0].astype(float).tolist())

        return actions

    def update(
        self,
        observations: list[list[float]],
        actions: list[list[float]],
        reward: list[float],
        next_observations: list[list[float]],
        done: bool,
    ) -> None:
        shared_context = build_shared_context(observations, self.observation_names)
        next_shared_context = build_shared_context(next_observations, self.observation_names)

        for index, (observation, action, reward_value, next_observation) in enumerate(
            zip(observations, actions, reward, next_observations)
        ):
            encoded_observation = self._encode_observation(index, observation, shared_context)
            encoded_next_observation = self._encode_observation(index, next_observation, next_shared_context)
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
            rewards_batch = np.asarray([item[2] for item in self.replay_buffer.buffer], dtype="float32")
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
                for sample_observation, sample_action, sample_reward, sample_next_observation, sample_done in self.replay_buffer.buffer
            ]
            self.normalized = True

        stats: list[dict[str, float]] = []
        for _ in range(self.update_per_time_step):
            sample_observation, sample_action, sample_reward, sample_next_observation, sample_done = self.replay_buffer.sample(
                self.batch_size
            )
            observation_tensor = _tensor(sample_observation, device=self.device)
            next_observation_tensor = _tensor(sample_next_observation, device=self.device)
            action_tensor = _tensor(sample_action, device=self.device)
            reward_tensor = _tensor(sample_reward, device=self.device).unsqueeze(1)
            done_tensor = _tensor(sample_done, device=self.device).unsqueeze(1)

            with torch.no_grad():
                next_action, next_log_pi, _ = self.policy_net.sample(next_observation_tensor)
                alpha_value = self._alpha_value()
                target_q = torch.min(
                    self.target_soft_q_net1(next_observation_tensor, next_action),
                    self.target_soft_q_net2(next_observation_tensor, next_action),
                ) - alpha_value * next_log_pi
                q_target = reward_tensor + (1.0 - done_tensor) * self.discount * target_q

            q1_prediction = self.soft_q_net1(observation_tensor, action_tensor)
            q2_prediction = self.soft_q_net2(observation_tensor, action_tensor)
            q1_loss = self.soft_q_criterion(q1_prediction, q_target)
            q2_loss = self.soft_q_criterion(q2_prediction, q_target)
            self.soft_q_optimizer1.zero_grad()
            q1_loss.backward()
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.zero_grad()
            q2_loss.backward()
            self.soft_q_optimizer2.step()

            new_action, log_pi, _ = self.policy_net.sample(observation_tensor)
            q_new_action = torch.min(
                self.soft_q_net1(observation_tensor, new_action),
                self.soft_q_net2(observation_tensor, new_action),
            )
            alpha_value = self._alpha_value()
            policy_loss = (alpha_value * log_pi - q_new_action).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            alpha_loss_value = 0.0
            if self.auto_entropy_tuning and self.log_alpha is not None and self.alpha_optimizer is not None:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.detach().cpu().item())
                alpha_value = self._alpha_value()

            for target_param, param in zip(
                self.target_soft_q_net1.parameters(),
                self.soft_q_net1.parameters(),
            ):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            for target_param, param in zip(
                self.target_soft_q_net2.parameters(),
                self.soft_q_net2.parameters(),
            ):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            stats.append(
                {
                    "q1_loss": float(q1_loss.detach().cpu().item()),
                    "q2_loss": float(q2_loss.detach().cpu().item()),
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "alpha": float(alpha_value.detach().cpu().item())
                    if isinstance(alpha_value, torch.Tensor)
                    else float(alpha_value),
                    "alpha_loss": alpha_loss_value,
                    "buffer_size": float(len(self.replay_buffer)),
                }
            )

        self.last_update_stats = {
            key: float(np.mean([item[key] for item in stats], dtype="float32"))
            for key in stats[0]
        }

    def training_stats(self) -> dict[str, float]:
        return dict(self.last_update_stats)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "controller_type": self.controller_type,
            "hidden_dimension": list(self.hidden_dimension),
            "discount": float(self.discount),
            "tau": float(self.tau),
            "alpha": float(self.alpha),
            "lr": float(self.lr),
            "batch_size": int(self.batch_size),
            "replay_buffer_capacity": int(self.replay_buffer_capacity),
            "standardize_start_time_step": int(self.standardize_start_time_step),
            "end_exploration_time_step": int(self.end_exploration_time_step),
            "action_scaling_coefficient": float(self.action_scaling_coefficient),
            "reward_scaling": float(self.reward_scaling),
            "update_per_time_step": int(self.update_per_time_step),
            "shared_context_dimension": int(self.shared_context_dimension),
            "auto_entropy_tuning": bool(self.auto_entropy_tuning),
            "time_step": int(self.time_step),
            "normalized": bool(self.normalized),
            "policy_state_dict": self.policy_net.state_dict(),
            "soft_q1_state_dict": self.soft_q_net1.state_dict(),
            "soft_q2_state_dict": self.soft_q_net2.state_dict(),
            "target_soft_q1_state_dict": self.target_soft_q_net1.state_dict(),
            "target_soft_q2_state_dict": self.target_soft_q_net2.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "soft_q_optimizer1_state_dict": self.soft_q_optimizer1.state_dict(),
            "soft_q_optimizer2_state_dict": self.soft_q_optimizer2.state_dict(),
            "log_alpha": None if self.log_alpha is None else float(self.log_alpha.detach().cpu().item()),
            "alpha_optimizer_state_dict": None
            if self.alpha_optimizer is None
            else self.alpha_optimizer.state_dict(),
            "norm_mean": _array_to_list(self.norm_mean),
            "norm_std": _array_to_list(self.norm_std),
            "r_norm_mean": self.r_norm_mean,
            "r_norm_std": self.r_norm_std,
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        restored_time_step = int(payload["time_step"])
        self._Environment__time_step = restored_time_step
        self._Agent__actions = [
            [[] for _ in range(restored_time_step + 1)] for _ in self.action_space
        ]
        self.normalized = bool(payload["normalized"])
        self.norm_mean = (
            None if payload["norm_mean"] is None else np.asarray(payload["norm_mean"], dtype="float32")
        )
        self.norm_std = (
            None if payload["norm_std"] is None else np.asarray(payload["norm_std"], dtype="float32")
        )
        self.r_norm_mean = (
            None if payload["r_norm_mean"] is None else float(payload["r_norm_mean"])
        )
        self.r_norm_std = None if payload["r_norm_std"] is None else float(payload["r_norm_std"])

        self.policy_net.load_state_dict(payload["policy_state_dict"])
        self.soft_q_net1.load_state_dict(payload["soft_q1_state_dict"])
        self.soft_q_net2.load_state_dict(payload["soft_q2_state_dict"])
        self.target_soft_q_net1.load_state_dict(payload["target_soft_q1_state_dict"])
        self.target_soft_q_net2.load_state_dict(payload["target_soft_q2_state_dict"])
        self.policy_optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
        self.soft_q_optimizer1.load_state_dict(payload["soft_q_optimizer1_state_dict"])
        self.soft_q_optimizer2.load_state_dict(payload["soft_q_optimizer2_state_dict"])

        if self.auto_entropy_tuning and payload.get("log_alpha") is not None:
            self.log_alpha = torch.tensor(
                float(payload["log_alpha"]),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            state_dict = payload.get("alpha_optimizer_state_dict")
            if state_dict is not None:
                self.alpha_optimizer.load_state_dict(state_dict)

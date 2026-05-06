from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn, optim

from cos435_citylearn.algorithms.mappo.features import (
    CENTRALIZED_CRITIC_CONTEXT_VERSION,
    build_centralized_critic_context,
    centralized_critic_context_dimension,
)
from cos435_citylearn.algorithms.mappo.rollout_buffer import CentralizedRolloutBuffer
from cos435_citylearn.algorithms.ppo.controllers import (
    SharedPPOController,
    _array_to_list,
    normalize_rollout_advantages,
)
from cos435_citylearn.algorithms.ppo.networks import ValueNetwork
from cos435_citylearn.algorithms.ppo.shared_features import (
    SHARED_CONTEXT_V2_DIMENSION,
    build_shared_context_v2,
)


class CentralizedMAPPOController(SharedPPOController):
    def __init__(
        self,
        env,
        *,
        critic_hidden_dimension: list[int] | None = None,
        critic_context_version: str = CENTRALIZED_CRITIC_CONTEXT_VERSION,
        normalize_critic_observations: bool = True,
        **kwargs,
    ) -> None:
        if critic_context_version != CENTRALIZED_CRITIC_CONTEXT_VERSION:
            raise ValueError(
                "critic_context_version must be "
                f"{CENTRALIZED_CRITIC_CONTEXT_VERSION!r} for MAPPO"
            )
        self.critic_context_version = critic_context_version
        self.normalize_critic_observations = bool(normalize_critic_observations)
        self.critic_hidden_dimension = list(
            critic_hidden_dimension
            if critic_hidden_dimension is not None
            else kwargs.get("hidden_dimension", [64, 64])
        )
        super().__init__(env, **kwargs)

        local_actor_dim = self._observation_dimension_cached - self.shared_context_dimension
        self.critic_context_dimension = centralized_critic_context_dimension(
            local_actor_dim,
            self.shared_context_dimension,
        )
        self._critic_observation_dimension_cached = (
            self._observation_dimension_cached + self.critic_context_dimension
        )
        self.value_net = ValueNetwork(
            self._critic_observation_dimension_cached,
            self.critic_hidden_dimension,
        ).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

        n_buildings = len(self.action_space)
        self.rollout_buffer = CentralizedRolloutBuffer(
            n_steps=self.rollout_steps,
            n_buildings=n_buildings,
            actor_observation_dim=self._observation_dimension_cached,
            critic_observation_dim=self._critic_observation_dimension_cached,
            action_dim=self._action_dimension_cached,
        )
        self.critic_obs_rms_mean: np.ndarray | None = None
        self.critic_obs_rms_var: np.ndarray | None = None
        self.critic_obs_rms_count: float = 1e-4

    @property
    def controller_type(self) -> str:
        return "forecast_augmented_mappo"

    def _encode_local_observation(
        self,
        index: int,
        observation: Sequence[float],
    ) -> np.ndarray:
        stacked = np.hstack(self.encoders[index] * np.asarray(observation, dtype="float32"))
        return np.asarray(
            [value for value in stacked if value is not None],
            dtype="float32",
        )

    def _encode_actor_and_critic(
        self,
        observations: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        shared_context = build_shared_context_v2(observations, self.observation_names)
        local_encoded = np.zeros(
            (
                len(observations),
                self._observation_dimension_cached - self.shared_context_dimension,
            ),
            dtype=np.float32,
        )
        for index, observation in enumerate(observations):
            local_encoded[index] = self._encode_local_observation(index, observation)

        actor_observations = np.zeros(
            (len(observations), self._observation_dimension_cached),
            dtype=np.float32,
        )
        for index in range(len(observations)):
            actor_observations[index] = np.concatenate(
                [local_encoded[index], shared_context],
                dtype=np.float32,
            )

        critic_context = build_centralized_critic_context(local_encoded, shared_context)
        critic_observations = np.zeros(
            (len(observations), self._critic_observation_dimension_cached),
            dtype=np.float32,
        )
        for index in range(len(observations)):
            critic_observations[index] = np.concatenate(
                [actor_observations[index], critic_context],
                dtype=np.float32,
            )
        return actor_observations, critic_observations

    def _update_critic_obs_rms(self, critic_obs: np.ndarray) -> None:
        if self.critic_obs_rms_mean is None:
            self.critic_obs_rms_mean = critic_obs.mean(axis=0)
            self.critic_obs_rms_var = critic_obs.var(axis=0) + 1e-8
            self.critic_obs_rms_count = float(critic_obs.shape[0])
            return

        batch_mean = critic_obs.mean(axis=0)
        batch_var = critic_obs.var(axis=0)
        batch_count = float(critic_obs.shape[0])

        delta = batch_mean - self.critic_obs_rms_mean
        total_count = self.critic_obs_rms_count + batch_count
        new_mean = self.critic_obs_rms_mean + delta * batch_count / total_count
        m_a = self.critic_obs_rms_var * self.critic_obs_rms_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.critic_obs_rms_count * batch_count / total_count
        self.critic_obs_rms_mean = new_mean
        self.critic_obs_rms_var = m2 / total_count + 1e-8
        self.critic_obs_rms_count = total_count

    def _normalize_critic_obs(self, critic_obs: np.ndarray) -> np.ndarray:
        if not self.normalize_critic_observations or self.critic_obs_rms_mean is None:
            return critic_obs
        normalized = (critic_obs - self.critic_obs_rms_mean) / np.sqrt(self.critic_obs_rms_var)
        return normalized.astype(np.float32)

    def sample_rollout_step(
        self,
        observations: list[list[float]],
    ) -> dict[str, np.ndarray]:
        actor_obs, critic_obs = self._encode_actor_and_critic(observations)
        if self.normalize_observations:
            self._update_obs_rms(actor_obs)
        if self.normalize_critic_observations:
            self._update_critic_obs_rms(critic_obs)
        normalized_actor = self._normalize_obs(actor_obs)
        normalized_critic = self._normalize_critic_obs(critic_obs)

        actor_tensor = torch.as_tensor(normalized_actor, dtype=torch.float32, device=self.device)
        critic_tensor = torch.as_tensor(normalized_critic, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_tensor, pre_tanh, log_prob = self.policy_net.sample(actor_tensor)
            value_tensor = self.value_net(critic_tensor)

        actions_array = action_tensor.detach().cpu().numpy().astype(np.float32)
        return {
            "encoded_observations": normalized_actor.astype(np.float32),
            "critic_observations": normalized_critic.astype(np.float32),
            "actions": actions_array,
            "pre_tanh": pre_tanh.detach().cpu().numpy().astype(np.float32),
            "log_probs": log_prob.detach().cpu().numpy().astype(np.float32),
            "values": value_tensor.detach().cpu().numpy().astype(np.float32),
            "actions_list": actions_array.tolist(),
        }

    def value(self, observations: list[list[float]]) -> np.ndarray:
        _, critic_obs = self._encode_actor_and_critic(observations)
        normalized_critic = self._normalize_critic_obs(critic_obs)
        critic_tensor = torch.as_tensor(normalized_critic, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value_tensor = self.value_net(critic_tensor)
        return value_tensor.detach().cpu().numpy().astype(np.float32)

    def store_rollout_step(
        self,
        *,
        step_payload: dict[str, np.ndarray],
        rewards: list[float],
        done: bool,
    ) -> np.ndarray:
        n = len(rewards)
        reward_array = np.asarray(rewards, dtype=np.float32)
        done_array = np.full(n, 1.0 if done else 0.0, dtype=np.float32)
        normalized_rewards = self._update_reward_rms(reward_array, done_array)

        self.rollout_buffer.add(
            actor_observations=step_payload["encoded_observations"],
            critic_observations=step_payload["critic_observations"],
            pre_tanh_actions=step_payload["pre_tanh"],
            actions=step_payload["actions"],
            log_probs=step_payload["log_probs"],
            values=step_payload["values"],
            rewards=normalized_rewards,
            dones=done_array,
        )
        if done:
            self.returns_running = np.zeros_like(self.returns_running)
        return normalized_rewards

    def finish_rollout(self, last_observations: list[list[float]]) -> dict[str, float]:
        last_values = self.value(last_observations)
        self.rollout_buffer.compute_gae(
            last_values=last_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        n_updates = 0
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []

        advantages_all = self.rollout_buffer.advantages[: self.rollout_buffer.size].reshape(-1)
        advantages_all = normalize_rollout_advantages(
            advantages_all,
            enabled=self.normalize_advantage,
        )
        self.rollout_buffer.advantages[: self.rollout_buffer.size] = advantages_all.reshape(
            self.rollout_buffer.size,
            self.rollout_buffer.n_buildings,
        )

        for _ in range(self.n_epochs):
            epoch_kls = []
            for batch in self.rollout_buffer.iter_minibatches(
                batch_size=self.minibatch_size,
                shuffle=True,
                device=self.device,
                normalize_advantage=self.normalize_advantage,
            ):
                new_log_prob, entropy = self.policy_net.evaluate_actions(
                    batch["actor_observations"],
                    batch["pre_tanh_actions"],
                )
                new_values = self.value_net(batch["critic_observations"])

                ratio = torch.exp(new_log_prob - batch["log_probs"])
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                unclipped_obj = ratio * batch["advantages"]
                clipped_obj = clipped_ratio * batch["advantages"]
                policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

                value_loss = 0.5 * (new_values - batch["returns"]).pow(2).mean()
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_prob - batch["log_probs"]
                    approx_kl_value = float(((log_ratio.exp() - 1) - log_ratio).mean().item())
                    clip_fraction_value = float(
                        (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                    )

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                approx_kls.append(approx_kl_value)
                epoch_kls.append(approx_kl_value)
                clip_fractions.append(clip_fraction_value)
                n_updates += 1

            if (
                self.target_kl is not None
                and epoch_kls
                and float(np.mean(epoch_kls)) > self.target_kl
            ):
                break

        self.rollout_buffer.reset()
        self._total_updates += n_updates

        def _mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        self.last_update_stats = {
            "policy_loss": _mean(policy_losses),
            "value_loss": _mean(value_losses),
            "entropy_loss": _mean(entropy_losses),
            "approx_kl": _mean(approx_kls),
            "clip_fraction": _mean(clip_fractions),
            "n_updates": float(n_updates),
            "total_updates": float(self._total_updates),
        }
        return self.last_update_stats

    def checkpoint_state(self) -> dict[str, Any]:
        payload = super().checkpoint_state()
        payload.update(
            {
                "controller_type": self.controller_type,
                "critic_hidden_dimension": list(self.critic_hidden_dimension),
                "critic_context_dimension": int(self.critic_context_dimension),
                "critic_context_version": self.critic_context_version,
                "normalize_critic_observations": bool(self.normalize_critic_observations),
                "critic_obs_rms_mean": _array_to_list(self.critic_obs_rms_mean),
                "critic_obs_rms_var": _array_to_list(self.critic_obs_rms_var),
                "critic_obs_rms_count": float(self.critic_obs_rms_count),
            }
        )
        return payload

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        super().load_checkpoint_state(payload)
        critic_obs_mean = payload.get("critic_obs_rms_mean")
        critic_obs_var = payload.get("critic_obs_rms_var")
        self.critic_obs_rms_mean = (
            None if critic_obs_mean is None else np.asarray(critic_obs_mean, dtype=np.float32)
        )
        self.critic_obs_rms_var = (
            None if critic_obs_var is None else np.asarray(critic_obs_var, dtype=np.float32)
        )
        self.critic_obs_rms_count = float(payload.get("critic_obs_rms_count", 1e-4))


__all__ = [
    "CENTRALIZED_CRITIC_CONTEXT_VERSION",
    "CentralizedMAPPOController",
    "SHARED_CONTEXT_V2_DIMENSION",
]

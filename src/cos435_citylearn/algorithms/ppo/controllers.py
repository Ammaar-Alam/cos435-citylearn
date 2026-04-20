from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from citylearn.agents.rlc import RLC
from citylearn.preprocessing import Encoder, RemoveFeature
from torch import nn, optim

from cos435_citylearn.algorithms.ppo.networks import ActorNetwork, ValueNetwork
from cos435_citylearn.algorithms.ppo.rollout_buffer import RolloutBuffer
from cos435_citylearn.algorithms.ppo.schedules import parse_ent_coef
from cos435_citylearn.algorithms.ppo.shared_features import (
    SHARED_CONTEXT_V2_DIMENSION,
    build_shared_context_v2,
)


def _array_to_list(values: Any) -> Any:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        return values.tolist()
    return values


def assert_minibatch_fits_rollout(
    *, minibatch_size: int, rollout_steps: int, n_buildings: int
) -> None:
    rollout_batch_size = int(rollout_steps) * int(n_buildings)
    if int(minibatch_size) > rollout_batch_size:
        # iter_minibatches clamps batch_size via max(1, min(batch_size, n)), so
        # an oversized minibatch is silently shrunk to the rollout batch size --
        # the configured minibatch_size is never honored. Fail loudly instead.
        raise ValueError(
            f"minibatch_size ({minibatch_size}) exceeds the rollout batch size "
            f"({rollout_batch_size} = rollout_steps {rollout_steps} * "
            f"n_buildings {n_buildings}); iter_minibatches would silently shrink "
            "the minibatch to the rollout batch size. "
            "reduce minibatch_size or increase rollout_steps."
        )


def normalize_rollout_advantages(
    advantages: np.ndarray, *, enabled: bool
) -> np.ndarray:
    """Zero-center and unit-variance scale a flat advantage array when enabled.

    PPO additionally normalizes per-minibatch inside
    ``RolloutBuffer.iter_minibatches`` when the same flag is set. Gating both
    passes on ``normalize_advantage`` makes the config knob a true on/off
    switch; ungated global normalization was a latent bug where
    ``normalize_advantage=False`` silently still produced standardized
    advantages because the global pass ran before the per-minibatch gate.
    """
    if not enabled or advantages.size <= 1:
        return advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    return (advantages - adv_mean) / adv_std


class SharedPPOController(RLC):
    def __init__(
        self,
        env,
        *,
        hidden_dimension: list[int] | None = None,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        minibatch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float | dict[str, float] = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 2048,
        reward_scaling: float = 1.0,
        shared_context_dimension: int = SHARED_CONTEXT_V2_DIMENSION,
        shared_context_version: str = "v2",
        normalize_observations: bool = True,
        normalize_rewards: bool = True,
        normalize_advantage: bool = True,
        target_kl: float | None = None,
    ) -> None:
        if shared_context_dimension != SHARED_CONTEXT_V2_DIMENSION:
            raise ValueError(
                f"shared_context_dimension must be {SHARED_CONTEXT_V2_DIMENSION} "
                "for shared PPO v2 context"
            )
        if shared_context_version != "v2":
            raise ValueError(
                f"shared_context_version must be 'v2' (got {shared_context_version!r}); "
                "v1 uses count-dependent positive_load_sum and is rejected for shared PPO."
            )

        self.shared_context_dimension = int(shared_context_dimension)
        self.shared_context_version = shared_context_version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_range = float(clip_range)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        initial_ent, self._ent_coef_schedule = parse_ent_coef(ent_coef)
        self.ent_coef = float(initial_ent)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.rollout_steps = int(rollout_steps)
        self.normalize_observations = bool(normalize_observations)
        self.normalize_rewards = bool(normalize_rewards)
        self.normalize_advantage = bool(normalize_advantage)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.last_update_stats: dict[str, float] = {}
        self._total_updates = 0

        # inherit from RLC so we get encoders + obs/action space for free
        super().__init__(
            env,
            hidden_dimension=hidden_dimension,
            discount=None,
            tau=None,
            alpha=None,
            lr=lr,
            batch_size=None,
            replay_buffer_capacity=None,
            standardize_start_time_step=None,
            end_exploration_time_step=None,
            action_scaling_coefficienct=None,
            reward_scaling=reward_scaling,
            update_per_time_step=None,
        )

        self._validate_shared_spaces()
        self._observation_dimension_cached = self._compute_observation_dimension()
        self._action_dimension_cached = int(self.action_dimension[0])

        action_low, action_high = self._action_bounds()
        self.policy_net = ActorNetwork(
            self._observation_dimension_cached,
            self._action_dimension_cached,
            list(self.hidden_dimension),
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.value_net = ValueNetwork(
            self._observation_dimension_cached,
            list(self.hidden_dimension),
        ).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

        n_buildings = len(self.action_space)
        assert_minibatch_fits_rollout(
            minibatch_size=self.minibatch_size,
            rollout_steps=self.rollout_steps,
            n_buildings=n_buildings,
        )

        self.rollout_buffer = RolloutBuffer(
            n_steps=self.rollout_steps,
            n_buildings=n_buildings,
            observation_dim=self._observation_dimension_cached,
            action_dim=self._action_dimension_cached,
        )

        self.obs_rms_mean: np.ndarray | None = None
        self.obs_rms_var: np.ndarray | None = None
        self.obs_rms_count: float = 1e-4
        self.reward_rms_var: float | None = None
        self.reward_rms_count: float = 1e-4
        self.reward_rms_mean: float = 0.0
        self.returns_running: np.ndarray = np.zeros(len(self.action_space), dtype=np.float32)

    @property
    def controller_type(self) -> str:
        return "shared_parameter_ppo"

    def set_training_progress(self, progress: float) -> None:
        if self._ent_coef_schedule is None:
            return
        self.ent_coef = float(self._ent_coef_schedule.value_at(progress))

    def _validate_shared_spaces(self) -> None:
        reference_action_shape = tuple(self.action_space[0].shape)
        reference_action_dim = self.action_dimension[0]
        for action_dimension, space in zip(self.action_dimension, self.action_space):
            if (
                action_dimension != reference_action_dim
                or tuple(space.shape) != reference_action_shape
            ):
                raise ValueError("shared PPO requires identical per-building action spaces")

    def set_encoders(self) -> list[list[Encoder]]:
        encoders = super().set_encoders()
        for index, names in enumerate(self.observation_names):
            for feature_index, name in enumerate(names):
                if name == "net_electricity_consumption":
                    encoders[index][feature_index] = RemoveFeature()
        return encoders

    def _compute_observation_dimension(self) -> int:
        encoded_lengths = []
        for encoders, space in zip(self.encoders, self.observation_space):
            stacked = np.hstack(encoders * np.ones(len(space.low)))
            encoded = [value for value in stacked if value is not None]
            encoded_lengths.append(len(encoded))
        if len(set(encoded_lengths)) != 1:
            raise ValueError("shared PPO requires identical per-building encoded observation sizes")
        return int(encoded_lengths[0]) + self.shared_context_dimension

    def _action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.as_tensor(self.action_space[0].low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(self.action_space[0].high, dtype=torch.float32, device=self.device)
        return low, high

    def _encode_observation(
        self,
        index: int,
        observation: Sequence[float],
        shared_context: np.ndarray,
    ) -> np.ndarray:
        stacked = np.hstack(
            self.encoders[index] * np.asarray(observation, dtype="float32")
        )
        encoded = np.asarray(
            [value for value in stacked if value is not None],
            dtype="float32",
        )
        return np.concatenate([encoded, shared_context], dtype="float32")

    def _encode_all(self, observations: Sequence[Sequence[float]]) -> np.ndarray:
        shared_context = build_shared_context_v2(observations, self.observation_names)
        encoded = np.zeros(
            (len(observations), self._observation_dimension_cached),
            dtype=np.float32,
        )
        for index, observation in enumerate(observations):
            encoded[index] = self._encode_observation(index, observation, shared_context)
        return encoded

    def _update_obs_rms(self, encoded_obs: np.ndarray) -> None:
        if self.obs_rms_mean is None:
            self.obs_rms_mean = encoded_obs.mean(axis=0)
            self.obs_rms_var = encoded_obs.var(axis=0) + 1e-8
            self.obs_rms_count = float(encoded_obs.shape[0])
            return

        batch_mean = encoded_obs.mean(axis=0)
        batch_var = encoded_obs.var(axis=0)
        batch_count = float(encoded_obs.shape[0])

        delta = batch_mean - self.obs_rms_mean
        total_count = self.obs_rms_count + batch_count
        new_mean = self.obs_rms_mean + delta * batch_count / total_count
        m_a = self.obs_rms_var * self.obs_rms_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.obs_rms_count * batch_count / total_count
        self.obs_rms_mean = new_mean
        self.obs_rms_var = m2 / total_count + 1e-8
        self.obs_rms_count = total_count

    def _normalize_obs(self, encoded_obs: np.ndarray) -> np.ndarray:
        if not self.normalize_observations or self.obs_rms_mean is None:
            return encoded_obs
        return ((encoded_obs - self.obs_rms_mean) / np.sqrt(self.obs_rms_var)).astype(np.float32)

    def _update_reward_rms(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        # scale by running std of discounted returns, don't center
        if not self.normalize_rewards:
            return rewards.astype(np.float32)
        updated_returns = self.returns_running * self.gamma * (1.0 - dones) + rewards
        self.returns_running = updated_returns

        batch_var = float(updated_returns.var())
        batch_count = float(updated_returns.shape[0])
        batch_mean = float(updated_returns.mean())

        if self.reward_rms_var is None:
            self.reward_rms_mean = batch_mean
            self.reward_rms_var = batch_var + 1e-8
            self.reward_rms_count = batch_count
        else:
            delta = batch_mean - self.reward_rms_mean
            total_count = self.reward_rms_count + batch_count
            new_mean = self.reward_rms_mean + delta * batch_count / total_count
            m_a = self.reward_rms_var * self.reward_rms_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + (delta * delta) * self.reward_rms_count * batch_count / total_count
            self.reward_rms_mean = new_mean
            self.reward_rms_var = m2 / total_count + 1e-8
            self.reward_rms_count = total_count

        std = float(np.sqrt(self.reward_rms_var))
        return (rewards / max(std, 1e-8)).astype(np.float32)

    def predict(
        self,
        observations: list[list[float]],
        deterministic: bool = False,
    ) -> list[list[float]]:
        encoded = self._encode_all(observations)
        normalized = self._normalize_obs(encoded)
        observation_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                action_tensor = self.policy_net.deterministic_action(observation_tensor)
            else:
                action_tensor, _, _ = self.policy_net.sample(observation_tensor)

        actions = action_tensor.detach().cpu().numpy().astype(float).tolist()
        self.actions = actions
        self.next_time_step()
        return actions

    def sample_rollout_step(
        self,
        observations: list[list[float]],
    ) -> dict[str, np.ndarray]:
        encoded = self._encode_all(observations)
        if self.normalize_observations:
            self._update_obs_rms(encoded)
        normalized = self._normalize_obs(encoded)
        observation_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_tensor, pre_tanh, log_prob = self.policy_net.sample(observation_tensor)
            value_tensor = self.value_net(observation_tensor)

        actions_array = action_tensor.detach().cpu().numpy().astype(np.float32)
        return {
            "encoded_observations": normalized.astype(np.float32),
            "actions": actions_array,
            "pre_tanh": pre_tanh.detach().cpu().numpy().astype(np.float32),
            "log_probs": log_prob.detach().cpu().numpy().astype(np.float32),
            "values": value_tensor.detach().cpu().numpy().astype(np.float32),
            "actions_list": actions_array.tolist(),
        }

    def value(self, observations: list[list[float]]) -> np.ndarray:
        encoded = self._encode_all(observations)
        normalized = self._normalize_obs(encoded)
        observation_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value_tensor = self.value_net(observation_tensor)
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
            observations=step_payload["encoded_observations"],
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

        # Gate BOTH the global pass here and the per-minibatch pass (inside
        # iter_minibatches below) on self.normalize_advantage so the config
        # knob is a real on/off switch. Previously the global pass ran
        # unconditionally, making normalize_advantage=False a no-op.
        advantages_all = self.rollout_buffer.advantages[: self.rollout_buffer.size].reshape(-1)
        advantages_all = normalize_rollout_advantages(
            advantages_all, enabled=self.normalize_advantage
        )
        self.rollout_buffer.advantages[: self.rollout_buffer.size] = advantages_all.reshape(
            self.rollout_buffer.size, self.rollout_buffer.n_buildings
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
                    batch["observations"],
                    batch["pre_tanh_actions"],
                )
                new_values = self.value_net(batch["observations"])

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

    def training_stats(self) -> dict[str, float]:
        return dict(self.last_update_stats)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "controller_type": self.controller_type,
            "hidden_dimension": list(self.hidden_dimension),
            "lr": float(self.lr),
            "clip_range": float(self.clip_range),
            "n_epochs": int(self.n_epochs),
            "minibatch_size": int(self.minibatch_size),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "ent_coef": float(self.ent_coef),
            "ent_coef_schedule": (
                self._ent_coef_schedule.as_mapping()
                if self._ent_coef_schedule is not None
                else None
            ),
            "vf_coef": float(self.vf_coef),
            "max_grad_norm": float(self.max_grad_norm),
            "rollout_steps": int(self.rollout_steps),
            "reward_scaling": float(self.reward_scaling),
            "shared_context_dimension": int(self.shared_context_dimension),
            "shared_context_version": self.shared_context_version,
            "normalize_observations": bool(self.normalize_observations),
            "normalize_rewards": bool(self.normalize_rewards),
            "normalize_advantage": bool(self.normalize_advantage),
            "target_kl": None if self.target_kl is None else float(self.target_kl),
            "time_step": int(self.time_step),
            "total_updates": int(self._total_updates),
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "obs_rms_mean": _array_to_list(self.obs_rms_mean),
            "obs_rms_var": _array_to_list(self.obs_rms_var),
            "obs_rms_count": float(self.obs_rms_count),
            "reward_rms_mean": float(self.reward_rms_mean),
            "reward_rms_var": None if self.reward_rms_var is None else float(self.reward_rms_var),
            "reward_rms_count": float(self.reward_rms_count),
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(payload["policy_state_dict"])
        self.value_net.load_state_dict(payload["value_state_dict"])
        self.policy_optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(payload["value_optimizer_state_dict"])

        obs_mean = payload.get("obs_rms_mean")
        obs_var = payload.get("obs_rms_var")
        self.obs_rms_mean = None if obs_mean is None else np.asarray(obs_mean, dtype=np.float32)
        self.obs_rms_var = None if obs_var is None else np.asarray(obs_var, dtype=np.float32)
        self.obs_rms_count = float(payload.get("obs_rms_count", 1e-4))
        self.reward_rms_mean = float(payload.get("reward_rms_mean", 0.0))
        reward_rms_var = payload.get("reward_rms_var")
        self.reward_rms_var = None if reward_rms_var is None else float(reward_rms_var)
        self.reward_rms_count = float(payload.get("reward_rms_count", 1e-4))
        self._total_updates = int(payload.get("total_updates", 0))
        self.returns_running = np.zeros(len(self.action_space), dtype=np.float32)

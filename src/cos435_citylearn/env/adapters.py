from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StepResult:
    observations: list[list[float]]
    rewards: list[float]
    terminated: bool
    info: dict[str, Any]


def _to_nested_float_list(values: Any) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


class BaseCityLearnAdapter:
    def __init__(self, env: Any):
        self.env = env

    @property
    def action_names(self) -> list[list[str]]:
        return self.env.action_names

    @property
    def observation_names(self) -> list[list[str]]:
        return self.env.observation_names

    @property
    def done(self) -> bool:
        return bool(getattr(self.env, "done", False))

    def reset(self) -> list[list[float]]:
        observations = self.env.reset()
        if isinstance(observations, tuple):
            observations = observations[0]

        return _to_nested_float_list(observations)

    def clip_actions(self, actions: list[list[float]]) -> list[list[float]]:
        clipped_actions: list[list[float]] = []

        for box, values in zip(self.env.action_space, actions):
            array = np.asarray(values, dtype="float32")
            clipped = np.clip(array, box.low, box.high)
            clipped_actions.append(clipped.astype(float).tolist())

        return clipped_actions

    def sample_action(self, seed: int | None = None) -> list[list[float]]:
        rng = np.random.default_rng(seed)
        actions = []

        for box in self.env.action_space:
            sample = rng.uniform(box.low, box.high).astype("float32")
            actions.append(sample.astype(float).tolist())

        return actions

    def action_bounds(self) -> list[dict[str, list[float]]]:
        bounds = []

        for box in self.env.action_space:
            bounds.append(
                {
                    "low": box.low.astype(float).tolist(),
                    "high": box.high.astype(float).tolist(),
                }
            )

        return bounds

    def step(self, actions: list[list[float]]) -> StepResult:
        clipped_actions = self.clip_actions(actions)
        outcome = self.env.step(clipped_actions)

        if len(outcome) == 4:
            observations, rewards, done, info = outcome
        elif len(outcome) == 5:
            observations, rewards, terminated, truncated, info = outcome
            done = bool(terminated or truncated)
        else:
            raise ValueError(f"unexpected CityLearn step output length: {len(outcome)}")

        return StepResult(
            observations=_to_nested_float_list(observations),
            rewards=[float(value) for value in rewards],
            terminated=bool(done),
            info={} if info is None else dict(info),
        )


class CentralizedEnvAdapter(BaseCityLearnAdapter):
    def __init__(self, env: Any):
        super().__init__(env)
        if not env.central_agent:
            raise ValueError("CentralizedEnvAdapter requires central_agent=True")


class PerBuildingEnvAdapter(BaseCityLearnAdapter):
    def __init__(self, env: Any):
        super().__init__(env)
        if env.central_agent:
            raise ValueError("PerBuildingEnvAdapter requires central_agent=False")

from __future__ import annotations

import math
from functools import partial
from typing import Mapping, Sequence

import numpy as np
from citylearn.reward_function import RewardFunction

OFFICIAL_CHALLENGE_WEIGHTS = {
    "comfort": 0.30,
    "carbon": 0.10,
    "ramping": 0.075,
    "load_factor": 0.075,
    "daily_peak": 0.075,
    "all_time_peak": 0.075,
    "outage_comfort": 0.15,
    "outage_unserved": 0.15,
}


def _positive(value: float) -> float:
    return max(float(value), 0.0)


def _log1p(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


class OfficialChallengeReward(RewardFunction):
    def __init__(
        self,
        env_metadata: Mapping[str, object] | None,
        *,
        version: str = "reward_v1",
        comfort_band: float = 1.0,
        **kwargs,
    ):
        # CityLearn constructs reward functions with env_metadata=None and then injects
        # the real metadata later, so the repo-local state needs an explicit reset hook.
        super().__init__(env_metadata, **kwargs)
        self.version = version
        self.comfort_band = float(comfort_band)
        self.reset()

    def reset(self) -> None:
        self._step_index = 0
        self._previous_positive_load = 0.0
        self._daily_positive_loads: list[float] = []
        self._all_time_peak = 0.0
        self._previous_storage_soc: list[float] | None = None
        self._previous_storage_delta: list[float] | None = None

    def _comfort_excess(
        self,
        observations: Sequence[Mapping[str, float]],
        *,
        outage_only: bool,
    ) -> float:
        numerator = 0.0
        denominator = 0.0

        for observation in observations:
            occupant_count = max(float(observation.get("occupant_count", 0.0)), 0.0)
            if occupant_count <= 0.0:
                continue

            is_outage = float(observation.get("power_outage", 0.0)) > 0.0
            if outage_only and not is_outage:
                continue
            if not outage_only and is_outage:
                continue

            indoor = float(observation.get("indoor_dry_bulb_temperature", 0.0))
            set_point = float(observation.get("indoor_dry_bulb_temperature_set_point", indoor))
            numerator += max(abs(indoor - set_point) - self.comfort_band, 0.0) * occupant_count
            denominator += occupant_count

        if denominator <= 0.0:
            return 0.0

        return numerator / denominator

    def _storage_penalties(self, observations: Sequence[Mapping[str, float]]) -> tuple[float, float]:
        storage_socs = [float(o.get("electrical_storage_soc", 0.0)) for o in observations]
        if self._previous_storage_soc is None:
            deltas = [0.0 for _ in storage_socs]
            smoothness_penalty = 0.0
            sign_flip_penalty = 0.0
        else:
            deltas = [current - previous for current, previous in zip(storage_socs, self._previous_storage_soc)]
            smoothness_penalty = float(np.mean(np.abs(deltas), dtype="float32")) if deltas else 0.0
            if self._previous_storage_delta is None:
                sign_flip_penalty = 0.0
            else:
                flips = [
                    1.0 if current_delta * previous_delta < 0.0 else 0.0
                    for current_delta, previous_delta in zip(deltas, self._previous_storage_delta)
                ]
                sign_flip_penalty = float(np.mean(flips, dtype="float32")) if flips else 0.0

        self._previous_storage_soc = storage_socs
        self._previous_storage_delta = deltas
        return smoothness_penalty, sign_flip_penalty

    def calculate(self, observations: Sequence[Mapping[str, float]]) -> list[float]:
        if self._step_index > 0 and self._step_index % 24 == 0:
            self._daily_positive_loads = []

        positive_loads = [_positive(observation.get("net_electricity_consumption", 0.0)) for observation in observations]
        district_positive_load = float(np.sum(positive_loads, dtype="float32"))
        carbon_term = _log1p(
            sum(
                _positive(observation.get("net_electricity_consumption", 0.0))
                * max(float(observation.get("carbon_intensity", 0.0)), 0.0)
                for observation in observations
            )
        )
        ramping_term = _log1p(abs(district_positive_load - self._previous_positive_load))
        self._daily_positive_loads.append(district_positive_load)
        current_day_peak = max(self._daily_positive_loads, default=0.0)
        current_day_mean = (
            float(np.mean(self._daily_positive_loads, dtype="float32"))
            if self._daily_positive_loads
            else 0.0
        )
        load_factor_gap = (
            max(0.0, 1.0 - (current_day_mean / current_day_peak))
            if current_day_peak > 1e-6
            else 0.0
        )
        daily_peak_term = _log1p(current_day_peak)
        self._all_time_peak = max(self._all_time_peak, district_positive_load)
        all_time_peak_term = _log1p(self._all_time_peak)
        comfort_term = self._comfort_excess(observations, outage_only=False)
        outage_comfort_term = self._comfort_excess(observations, outage_only=True)
        outage_unserved_term = _log1p(
            sum(
                _positive(observation.get("net_electricity_consumption", 0.0))
                for observation in observations
                if float(observation.get("power_outage", 0.0)) > 0.0
            )
        )

        penalty = (
            OFFICIAL_CHALLENGE_WEIGHTS["comfort"] * comfort_term
            + OFFICIAL_CHALLENGE_WEIGHTS["carbon"] * carbon_term
            + OFFICIAL_CHALLENGE_WEIGHTS["ramping"] * ramping_term
            + OFFICIAL_CHALLENGE_WEIGHTS["load_factor"] * load_factor_gap
            + OFFICIAL_CHALLENGE_WEIGHTS["daily_peak"] * daily_peak_term
            + OFFICIAL_CHALLENGE_WEIGHTS["all_time_peak"] * all_time_peak_term
            + OFFICIAL_CHALLENGE_WEIGHTS["outage_comfort"] * outage_comfort_term
            + OFFICIAL_CHALLENGE_WEIGHTS["outage_unserved"] * outage_unserved_term
        )

        if self.version in {"reward_v2", "reward_v3"}:
            smoothness_penalty, sign_flip_penalty = self._storage_penalties(observations)
            penalty += 0.05 * smoothness_penalty
            penalty += 0.05 * sign_flip_penalty
        else:
            self._storage_penalties(observations)

        if self.version == "reward_v3":
            penalty += 0.10 * comfort_term
            penalty += 0.10 * outage_comfort_term
            penalty += 0.10 * outage_unserved_term

        self._previous_positive_load = district_positive_load
        self._step_index += 1
        reward_value = -float(penalty)

        if self.central_agent:
            return [reward_value]

        return [reward_value for _ in observations]


def resolve_reward_function(version: str):
    if version == "reward_v0":
        return None

    if version not in {"reward_v1", "reward_v2", "reward_v3"}:
        raise ValueError(f"unsupported SAC reward version: {version}")

    return partial(OfficialChallengeReward, version=version, comfort_band=1.0)

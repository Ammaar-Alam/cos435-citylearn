from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

SUPPORTED_EXPERT_POLICIES = {"basic_rbc", "adaptive_storage_v1"}


def _feature(
    names: Sequence[str],
    values: Sequence[float],
    name: str,
    default: float = 0.0,
) -> float:
    try:
        return float(values[names.index(name)])
    except (ValueError, IndexError, TypeError):
        return float(default)


def _hour(names: Sequence[str], values: Sequence[float]) -> int:
    raw_hour = _feature(names, values, "hour", default=0.0)
    return int(round(raw_hour))


def _forecast_values(names: Sequence[str], values: Sequence[float], base_name: str) -> list[float]:
    candidates = [
        _feature(names, values, base_name, default=np.nan),
        _feature(names, values, f"{base_name}_predicted_6h", default=np.nan),
        _feature(names, values, f"{base_name}_predicted_12h", default=np.nan),
        _feature(names, values, f"{base_name}_predicted_24h", default=np.nan),
    ]
    return [float(value) for value in candidates if np.isfinite(value)]


def _relative_signal(current: float, forecast: Sequence[float]) -> float:
    if not forecast:
        return 0.0
    center = float(np.median(np.asarray(forecast, dtype="float32")))
    spread = float(np.std(np.asarray(forecast, dtype="float32"))) + 1e-6
    return float((current - center) / spread)


def _basic_rbc_value(action_name: str, hour: int) -> float:
    if "storage" in action_name:
        if 9 <= hour <= 21:
            return -0.08
        if (1 <= hour <= 8) or (22 <= hour <= 24):
            return 0.091
        return 0.0

    if action_name == "cooling_device":
        if 9 <= hour <= 21:
            return 0.8
        if (1 <= hour <= 8) or (22 <= hour <= 24):
            return 0.4
        return 0.0

    if action_name == "heating_device":
        if 9 <= hour <= 21:
            return 0.4
        if (1 <= hour <= 8) or (22 <= hour <= 24):
            return 0.8
        return 0.0

    return 0.0


@dataclass(frozen=True)
class ExpertActionPolicy:
    policy_name: str
    observation_names: Sequence[Sequence[str]]
    action_names: Sequence[Sequence[str]]
    action_lows: Sequence[Sequence[float]] | None = None
    action_highs: Sequence[Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if self.policy_name not in SUPPORTED_EXPERT_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_EXPERT_POLICIES))
            raise ValueError(
                f"unsupported SAC expert policy '{self.policy_name}'; use one of {supported}"
            )

    def predict(self, observations: Sequence[Sequence[float]]) -> list[list[float]]:
        actions: list[list[float]] = []

        for index, observation in enumerate(observations):
            names = self.observation_names[index]
            action_names = self.action_names[index]
            if self.policy_name == "basic_rbc":
                row = self._basic_rbc_row(names, observation, action_names)
            else:
                row = self._adaptive_row(names, observation, action_names)
            actions.append(self._clip_row(index, row))

        return actions

    def metadata(self) -> dict[str, str]:
        return {"policy_name": self.policy_name}

    def _basic_rbc_row(
        self,
        names: Sequence[str],
        observation: Sequence[float],
        action_names: Sequence[str],
    ) -> list[float]:
        hour = _hour(names, observation)
        return [_basic_rbc_value(action_name, hour) for action_name in action_names]

    def _adaptive_row(
        self,
        names: Sequence[str],
        observation: Sequence[float],
        action_names: Sequence[str],
    ) -> list[float]:
        hour = _hour(names, observation)
        prices = _forecast_values(names, observation, "electricity_pricing")
        carbons = _forecast_values(names, observation, "carbon_intensity")
        current_price = _feature(names, observation, "electricity_pricing", default=0.0)
        current_carbon = _feature(names, observation, "carbon_intensity", default=0.0)
        price_signal = _relative_signal(current_price, prices)
        carbon_signal = _relative_signal(current_carbon, carbons)
        solar = _feature(names, observation, "solar_generation", default=0.0)
        load = _feature(names, observation, "non_shiftable_load", default=0.0)
        cooling_demand = _feature(names, observation, "cooling_demand", default=0.0)
        dhw_demand = _feature(names, observation, "dhw_demand", default=0.0)
        outage = _feature(names, observation, "power_outage", default=0.0) > 0.5
        row: list[float] = []

        for action_name in action_names:
            if action_name == "electrical_storage":
                value = self._electrical_storage_action(
                    names=names,
                    observation=observation,
                    hour=hour,
                    price_signal=price_signal,
                    carbon_signal=carbon_signal,
                    solar=solar,
                    load=load,
                    outage=outage,
                )
            elif action_name == "dhw_storage":
                value = self._thermal_storage_action(
                    names=names,
                    observation=observation,
                    soc_name="dhw_storage_soc",
                    demand=dhw_demand,
                    hour=hour,
                    price_signal=price_signal,
                    carbon_signal=carbon_signal,
                )
            elif action_name == "cooling_device":
                value = self._cooling_device_action(
                    names=names,
                    observation=observation,
                    cooling_demand=cooling_demand,
                    price_signal=price_signal,
                    carbon_signal=carbon_signal,
                    solar=solar,
                    hour=hour,
                )
            elif "storage" in action_name:
                value = _basic_rbc_value(action_name, hour)
            else:
                value = 0.0
            row.append(value)

        return row

    def _electrical_storage_action(
        self,
        *,
        names: Sequence[str],
        observation: Sequence[float],
        hour: int,
        price_signal: float,
        carbon_signal: float,
        solar: float,
        load: float,
        outage: bool,
    ) -> float:
        if outage:
            return 0.0

        soc = _feature(names, observation, "electrical_storage_soc", default=0.5)
        low_grid_signal = price_signal < -0.35 or carbon_signal < -0.35
        high_grid_signal = price_signal > 0.35 or carbon_signal > 0.35
        daytime_solar = solar > 0.08 and 8 <= hour <= 17
        evening_peak = 17 <= hour <= 22

        if soc < 0.18:
            return 0.45
        if (daytime_solar or low_grid_signal) and soc < 0.88:
            return 0.38 if load < 1.2 else 0.24
        if (evening_peak or high_grid_signal) and soc > 0.25:
            return -0.48 if load > 0.75 else -0.32
        if 1 <= hour <= 6 and soc < 0.65:
            return 0.22
        return 0.0

    def _thermal_storage_action(
        self,
        *,
        names: Sequence[str],
        observation: Sequence[float],
        soc_name: str,
        demand: float,
        hour: int,
        price_signal: float,
        carbon_signal: float,
    ) -> float:
        soc = _feature(names, observation, soc_name, default=0.5)
        if soc < 0.25:
            return 0.34
        if (price_signal < -0.25 or carbon_signal < -0.25 or 1 <= hour <= 7) and soc < 0.85:
            return 0.24
        if (price_signal > 0.35 or 16 <= hour <= 22) and soc > 0.30:
            return -0.26 if demand > 0.02 else -0.18
        return 0.0

    def _cooling_device_action(
        self,
        *,
        names: Sequence[str],
        observation: Sequence[float],
        cooling_demand: float,
        price_signal: float,
        carbon_signal: float,
        solar: float,
        hour: int,
    ) -> float:
        indoor = _feature(names, observation, "indoor_dry_bulb_temperature", default=24.0)
        set_point = _feature(
            names,
            observation,
            "indoor_dry_bulb_temperature_set_point",
            default=24.0,
        )
        hot_room = indoor > set_point + 0.4
        cheap_or_solar = price_signal < -0.25 or carbon_signal < -0.25 or solar > 0.08

        if cooling_demand <= 1e-6:
            return 0.35 if cheap_or_solar and 10 <= hour <= 17 else 0.2
        if hot_room:
            return 0.85
        if cheap_or_solar and 9 <= hour <= 17:
            return 0.72
        if price_signal > 0.4 or carbon_signal > 0.4:
            return 0.45
        return 0.62

    def _clip_row(self, index: int, row: Sequence[float]) -> list[float]:
        values = np.asarray(row, dtype="float32")
        if self.action_lows is None or self.action_highs is None:
            return values.astype(float).tolist()

        low = np.asarray(self.action_lows[index], dtype="float32")
        high = np.asarray(self.action_highs[index], dtype="float32")
        return np.clip(values, low, high).astype(float).tolist()

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class BuildingObservation:
    hour: float
    carbon_intensity: float
    electricity_price: float
    electricity_price_predicted_6h: float
    indoor_temperature: float
    non_shiftable_load: float
    solar_generation: float
    dhw_storage_soc: float
    electrical_storage_soc: float
    net_electricity_consumption: float
    cooling_demand: float
    dhw_demand: float
    occupant_count: float
    temperature_set_point: float
    power_outage: float


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _all_indices(names: Sequence[str]) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        indices[name].append(index)
    return dict(indices)


class ChescaLiteController:
    """CHESCA-inspired deterministic controller for CityLearn central-agent runs.

    The winning CHESCA controller combined comfort-preserving local controllers
    with community-level smoothing. This lightweight version keeps the same
    structure with only current/predicted observations available in the repo:
    comfort-first cooling, price/solar/load-aware storage, and a district-load
    smoothing correction.
    """

    def __init__(
        self,
        env: Any,
        *,
        cooling_scale: float = 0.6,
        cooling_bias: float = 0.0,
        cooling_max_delta: float = 0.12,
        storage_max_delta: float = 0.07,
        electrical_charge_scale: float = 1.0,
        electrical_discharge_scale: float = 1.0,
        outage_electrical_discharge: float = -0.55,
    ):
        if not bool(getattr(env, "central_agent", False)):
            raise ValueError("ChescaLiteController expects a central-agent CityLearn env")

        self.env = env
        self.cooling_scale = float(cooling_scale)
        self.cooling_bias = float(cooling_bias)
        self.cooling_max_delta = float(cooling_max_delta)
        self.storage_max_delta = float(storage_max_delta)
        self.electrical_charge_scale = float(electrical_charge_scale)
        self.electrical_discharge_scale = float(electrical_discharge_scale)
        self.outage_electrical_discharge = float(outage_electrical_discharge)
        self.observation_names = list(env.observation_names[0])
        self.action_names = list(env.action_names[0])
        self.observation_indices = _all_indices(self.observation_names)
        self.load_history: deque[float] = deque(maxlen=24)
        self.price_history: deque[float] = deque(maxlen=24)
        self.carbon_history: deque[float] = deque(maxlen=24)
        self.previous_action_vector: list[float] | None = None

    def predict(self, observations: list[list[float]]) -> list[list[float]]:
        if len(observations) != 1:
            raise ValueError("central-agent observations must contain one district vector")

        values = observations[0]
        building_count = self._building_count()
        building_observations = [
            self._building_observation(values, building_index)
            for building_index in range(building_count)
        ]
        district_positive_load = sum(
            max(item.net_electricity_consumption, 0.0) for item in building_observations
        )
        mean_price = float(
            np.mean([item.electricity_price for item in building_observations], dtype="float32")
        )
        mean_carbon = float(
            np.mean([item.carbon_intensity for item in building_observations], dtype="float32")
        )
        self.load_history.append(district_positive_load)
        self.price_history.append(mean_price)
        self.carbon_history.append(mean_carbon)

        action_occurrences: dict[str, int] = defaultdict(int)
        action_vector: list[float] = []

        for action_index, action_name in enumerate(self.action_names):
            building_index = action_occurrences[action_name]
            action_occurrences[action_name] += 1
            building = building_observations[min(building_index, building_count - 1)]

            if action_name == "dhw_storage":
                value = self._dhw_storage_action(building)
            elif action_name == "electrical_storage":
                value = self._electrical_storage_action(
                    building,
                    district_positive_load=district_positive_load,
                )
            elif action_name == "cooling_device":
                value = self._cooling_action(building)
            else:
                value = 0.0

            action_vector.append(float(value))

        smoothed_vector = self._smooth_actions(action_vector)
        self.previous_action_vector = list(smoothed_vector)
        return [smoothed_vector]

    def _building_count(self) -> int:
        return max(self.action_names.count("cooling_device"), 1)

    def _value(
        self,
        observation: Sequence[float],
        name: str,
        *,
        building_index: int | None = None,
        default: float = 0.0,
    ) -> float:
        indices = self.observation_indices.get(name)
        if not indices:
            return default

        if building_index is None:
            index = indices[0]
        else:
            index = indices[min(building_index, len(indices) - 1)]

        return float(observation[index])

    def _building_observation(
        self, observation: Sequence[float], building_index: int
    ) -> BuildingObservation:
        return BuildingObservation(
            hour=self._value(observation, "hour", default=1.0),
            carbon_intensity=self._value(observation, "carbon_intensity", default=0.0),
            electricity_price=self._value(observation, "electricity_pricing", default=0.0),
            electricity_price_predicted_6h=self._value(
                observation, "electricity_pricing_predicted_6h", default=0.0
            ),
            indoor_temperature=self._value(
                observation,
                "indoor_dry_bulb_temperature",
                building_index=building_index,
                default=24.0,
            ),
            non_shiftable_load=self._value(
                observation,
                "non_shiftable_load",
                building_index=building_index,
                default=0.0,
            ),
            solar_generation=self._value(
                observation, "solar_generation", building_index=building_index, default=0.0
            ),
            dhw_storage_soc=self._value(
                observation, "dhw_storage_soc", building_index=building_index, default=0.5
            ),
            electrical_storage_soc=self._value(
                observation,
                "electrical_storage_soc",
                building_index=building_index,
                default=0.5,
            ),
            net_electricity_consumption=self._value(
                observation,
                "net_electricity_consumption",
                building_index=building_index,
                default=0.0,
            ),
            cooling_demand=self._value(
                observation, "cooling_demand", building_index=building_index, default=0.0
            ),
            dhw_demand=self._value(
                observation, "dhw_demand", building_index=building_index, default=0.0
            ),
            occupant_count=self._value(
                observation, "occupant_count", building_index=building_index, default=0.0
            ),
            temperature_set_point=self._value(
                observation,
                "indoor_dry_bulb_temperature_set_point",
                building_index=building_index,
                default=24.0,
            ),
            power_outage=self._value(
                observation, "power_outage", building_index=building_index, default=0.0
            ),
        )

    def _dhw_storage_action(self, building: BuildingObservation) -> float:
        if building.power_outage > 0.0:
            return -0.10 if building.dhw_storage_soc > 0.25 and building.dhw_demand > 0 else 0.0

        if building.dhw_storage_soc < 0.18:
            return 0.18
        if building.dhw_demand > 0.05 and building.dhw_storage_soc > 0.28:
            return -0.05
        if building.hour >= 22 or building.hour <= 8:
            return 0.11 if building.dhw_storage_soc < 0.86 else 0.0
        if building.dhw_storage_soc > 0.72:
            return -0.04
        return 0.0

    def _electrical_storage_action(
        self,
        building: BuildingObservation,
        *,
        district_positive_load: float,
    ) -> float:
        if building.power_outage > 0.0:
            if building.electrical_storage_soc > 0.20:
                return self.outage_electrical_discharge
            return 0.0

        load_reference = max(float(np.mean(self.load_history, dtype="float32")), 1e-6)
        price_reference = max(float(np.mean(self.price_history, dtype="float32")), 1e-6)
        carbon_reference = max(float(np.mean(self.carbon_history, dtype="float32")), 1e-6)
        high_load = district_positive_load > 1.10 * load_reference
        low_load = district_positive_load < 0.75 * load_reference
        high_price = building.electricity_price > 1.05 * price_reference
        future_price_higher = (
            building.electricity_price_predicted_6h > building.electricity_price * 1.05
        )
        high_carbon = building.carbon_intensity > 1.05 * carbon_reference
        solar_surplus = building.solar_generation > 0.50 * max(building.non_shiftable_load, 1e-6)
        evening_peak = 16 <= int(building.hour) <= 21
        night_charge = int(building.hour) >= 22 or int(building.hour) <= 7

        if building.electrical_storage_soc < 0.16:
            return 0.14 * self.electrical_charge_scale
        if building.electrical_storage_soc > 0.82 and (high_load or evening_peak):
            return -0.26 * self.electrical_discharge_scale
        if solar_surplus and building.electrical_storage_soc < 0.88:
            return 0.20 * self.electrical_charge_scale
        if (
            night_charge or low_load or future_price_higher
        ) and building.electrical_storage_soc < 0.78:
            return 0.12 * self.electrical_charge_scale
        if (
            high_load or high_price or high_carbon or evening_peak
        ) and building.electrical_storage_soc > 0.28:
            return -0.18 * self.electrical_discharge_scale
        if building.net_electricity_consumption < 0.0 and building.electrical_storage_soc < 0.88:
            return 0.08 * self.electrical_charge_scale
        return 0.0

    def _cooling_action(self, building: BuildingObservation) -> float:
        occupied = building.occupant_count > 0.0
        comfort_error = building.indoor_temperature - building.temperature_set_point

        if building.power_outage > 0.0:
            return self._scaled_cooling(0.75 if occupied and comfort_error > 0.6 else 0.18)
        if not occupied:
            if comfort_error > 1.5:
                return self._scaled_cooling(0.35)
            if comfort_error > 0.5:
                return self._scaled_cooling(0.22)
            return self._scaled_cooling(0.05)
        if comfort_error > 2.0:
            return self._scaled_cooling(0.78)
        if comfort_error > 1.2:
            return self._scaled_cooling(0.62)
        if comfort_error > 0.6:
            return self._scaled_cooling(0.46)
        if comfort_error > 0.1:
            return self._scaled_cooling(0.30)
        if comfort_error < -0.4:
            return self._scaled_cooling(0.04)
        return self._scaled_cooling(0.16)

    def _scaled_cooling(self, value: float) -> float:
        return max(value * self.cooling_scale + self.cooling_bias, 0.0)

    def _smooth_actions(self, raw_actions: Sequence[float]) -> list[float]:
        smoothed: list[float] = []

        for action_index, (action_name, value) in enumerate(zip(self.action_names, raw_actions)):
            low = float(self.env.action_space[0].low[action_index])
            high = float(self.env.action_space[0].high[action_index])
            clipped = _clip(value, low, high)

            if self.previous_action_vector is None:
                smoothed.append(clipped)
                continue

            previous = float(self.previous_action_vector[action_index])
            if action_name == "cooling_device":
                max_delta = self.cooling_max_delta
            elif action_name in {"dhw_storage", "electrical_storage"}:
                # Keep outage dispatch ramp-limited too: fast outage-storage
                # screens reduced one unserved term but worsened released score.
                max_delta = self.storage_max_delta
            else:
                max_delta = high - low

            smoothed.append(_clip(clipped, previous - max_delta, previous + max_delta))

        return smoothed

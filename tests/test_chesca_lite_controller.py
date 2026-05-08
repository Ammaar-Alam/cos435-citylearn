from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cos435_citylearn.algorithms.heuristics import ChescaLiteController


def _fake_env() -> SimpleNamespace:
    observation_names = [
        [
            "day_type",
            "hour",
            "outdoor_dry_bulb_temperature",
            "outdoor_dry_bulb_temperature_predicted_6h",
            "outdoor_dry_bulb_temperature_predicted_12h",
            "outdoor_dry_bulb_temperature_predicted_24h",
            "diffuse_solar_irradiance",
            "diffuse_solar_irradiance_predicted_6h",
            "diffuse_solar_irradiance_predicted_12h",
            "diffuse_solar_irradiance_predicted_24h",
            "direct_solar_irradiance",
            "direct_solar_irradiance_predicted_6h",
            "direct_solar_irradiance_predicted_12h",
            "direct_solar_irradiance_predicted_24h",
            "carbon_intensity",
            "indoor_dry_bulb_temperature",
            "non_shiftable_load",
            "solar_generation",
            "dhw_storage_soc",
            "electrical_storage_soc",
            "net_electricity_consumption",
            "electricity_pricing",
            "electricity_pricing_predicted_6h",
            "electricity_pricing_predicted_12h",
            "electricity_pricing_predicted_24h",
            "cooling_demand",
            "dhw_demand",
            "occupant_count",
            "indoor_dry_bulb_temperature_set_point",
            "power_outage",
            "indoor_dry_bulb_temperature",
            "non_shiftable_load",
            "solar_generation",
            "dhw_storage_soc",
            "electrical_storage_soc",
            "net_electricity_consumption",
            "cooling_demand",
            "dhw_demand",
            "occupant_count",
            "indoor_dry_bulb_temperature_set_point",
            "power_outage",
        ]
    ]
    action_names = [
        [
            "dhw_storage",
            "electrical_storage",
            "cooling_device",
            "dhw_storage",
            "electrical_storage",
            "cooling_device",
        ]
    ]
    action_space = [
        SimpleNamespace(
            low=np.asarray([-1.0, -0.8, 0.0, -1.0, -0.4, 0.0], dtype="float32"),
            high=np.asarray([1.0, 0.8, 1.0, 1.0, 0.4, 1.0], dtype="float32"),
        )
    ]
    return SimpleNamespace(
        central_agent=True,
        observation_names=observation_names,
        action_names=action_names,
        action_space=action_space,
    )


def _observation(*, hour: float, indoor: float, set_point: float, net_load: float) -> list[float]:
    values = [0.0] * 41
    values[1] = hour
    values[14] = 0.5
    values[15] = indoor
    values[18] = 0.3
    values[19] = 0.5
    values[20] = net_load
    values[21] = 0.2
    values[22] = 0.2
    values[23] = 0.3
    values[24] = 0.4
    values[27] = 1.0
    values[28] = set_point
    values[30] = indoor
    values[33] = 0.3
    values[34] = 0.5
    values[35] = net_load
    values[38] = 1.0
    values[39] = set_point
    return values


def test_chesca_lite_returns_one_central_action_vector() -> None:
    controller = ChescaLiteController(_fake_env())

    actions = controller.predict(
        [_observation(hour=15.0, indoor=25.0, set_point=24.0, net_load=2.0)]
    )

    assert len(actions) == 1
    assert len(actions[0]) == 6
    assert all(-1.0 <= value <= 1.0 for value in actions[0])


def test_chesca_lite_cooling_tracks_occupied_comfort_error() -> None:
    controller = ChescaLiteController(_fake_env())

    warm = controller.predict(
        [_observation(hour=15.0, indoor=26.0, set_point=24.0, net_load=1.0)]
    )[0]
    comfortable = controller.predict(
        [_observation(hour=15.0, indoor=23.8, set_point=24.0, net_load=1.0)]
    )[0]

    assert warm[2] > comfortable[2]
    assert warm[5] > comfortable[5]


def test_chesca_lite_reads_cooling_set_point_alias() -> None:
    env = _fake_env()
    env.observation_names[0][28] = "indoor_dry_bulb_temperature_cooling_set_point"
    env.observation_names[0][39] = "indoor_dry_bulb_temperature_cooling_set_point"
    controller = ChescaLiteController(env)

    action = controller.predict(
        [_observation(hour=15.0, indoor=25.0, set_point=26.0, net_load=1.0)]
    )[0]

    assert action[2] < 0.1
    assert action[5] < 0.1


def test_chesca_lite_discharges_electrical_storage_on_high_load() -> None:
    controller = ChescaLiteController(_fake_env())

    low_load = controller.predict(
        [_observation(hour=3.0, indoor=23.8, set_point=24.0, net_load=-0.5)]
    )[0]
    high_load = controller.predict(
        [_observation(hour=18.0, indoor=23.8, set_point=24.0, net_load=3.0)]
    )[0]

    assert high_load[1] < low_load[1]
    assert high_load[4] < low_load[4]


def test_chesca_lite_accepts_controller_tuning_kwargs() -> None:
    controller = ChescaLiteController(
        _fake_env(),
        cooling_scale=0.5,
        cooling_bias=0.1,
        cooling_max_delta=0.2,
        storage_max_delta=0.1,
        electrical_charge_scale=0.8,
        electrical_discharge_scale=1.2,
        outage_electrical_discharge=-0.6,
    )

    action = controller.predict(
        [_observation(hour=15.0, indoor=26.0, set_point=24.0, net_load=1.0)]
    )[0]

    assert action[2] < 0.78
    assert action[5] < 0.78

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cos435_citylearn.algorithms.sac.controllers import ResidualSharedSACController
from cos435_citylearn.algorithms.sac.expert import ExpertActionPolicy

OBSERVATION_NAMES = [
    [
        "hour",
        "carbon_intensity",
        "non_shiftable_load",
        "solar_generation",
        "dhw_storage_soc",
        "electrical_storage_soc",
        "electricity_pricing",
        "electricity_pricing_predicted_6h",
        "electricity_pricing_predicted_12h",
        "electricity_pricing_predicted_24h",
        "cooling_demand",
        "dhw_demand",
        "indoor_dry_bulb_temperature",
        "indoor_dry_bulb_temperature_set_point",
        "power_outage",
    ]
]
ACTION_NAMES = [["dhw_storage", "electrical_storage", "cooling_device"]]
ACTION_LOWS = [[-1.0, -0.83, 0.0]]
ACTION_HIGHS = [[1.0, 0.83, 1.0]]


def test_basic_rbc_policy_matches_citylearn_hour_map() -> None:
    policy = ExpertActionPolicy(
        "basic_rbc",
        observation_names=OBSERVATION_NAMES,
        action_names=ACTION_NAMES,
        action_lows=ACTION_LOWS,
        action_highs=ACTION_HIGHS,
    )

    off_peak = [
        [22, 0.4, 0.3, 0.0, 0.5, 0.5, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0, 24.0, 24.0, 0]
    ]
    peak = [
        [12, 0.4, 0.3, 0.0, 0.5, 0.5, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0, 24.0, 24.0, 0]
    ]

    np.testing.assert_allclose(policy.predict(off_peak)[0], [0.091, 0.091, 0.4])
    np.testing.assert_allclose(policy.predict(peak)[0], [-0.08, -0.08, 0.8])


def test_adaptive_policy_discharges_electrical_storage_on_high_evening_price() -> None:
    policy = ExpertActionPolicy(
        "adaptive_storage_v1",
        observation_names=OBSERVATION_NAMES,
        action_names=ACTION_NAMES,
        action_lows=ACTION_LOWS,
        action_highs=ACTION_HIGHS,
    )

    actions = policy.predict(
        [
            [
                18,
                0.7,
                1.4,
                0.0,
                0.6,
                0.7,
                0.22,
                0.04,
                0.05,
                0.04,
                0.6,
                0.08,
                24.0,
                24.0,
                0,
            ]
        ]
    )

    assert actions[0][1] < 0.0


def test_adaptive_policy_charges_electrical_storage_with_solar_and_low_soc() -> None:
    policy = ExpertActionPolicy(
        "adaptive_storage_v1",
        observation_names=OBSERVATION_NAMES,
        action_names=ACTION_NAMES,
        action_lows=ACTION_LOWS,
        action_highs=ACTION_HIGHS,
    )

    actions = policy.predict(
        [
            [
                13,
                0.2,
                0.5,
                0.3,
                0.6,
                0.15,
                0.03,
                0.04,
                0.08,
                0.09,
                0.3,
                0.05,
                23.0,
                24.0,
                0,
            ]
        ]
    )

    assert actions[0][1] > 0.0


def test_expert_policy_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unsupported SAC expert policy"):
        ExpertActionPolicy(
            "does_not_exist",
            observation_names=OBSERVATION_NAMES,
            action_names=ACTION_NAMES,
        )


def test_residual_controller_composes_policy_delta_around_expert_action() -> None:
    controller = ResidualSharedSACController.__new__(ResidualSharedSACController)
    controller.action_scaling_coefficient = 0.5
    controller.residual_scaling_coefficient = 0.75
    controller.action_space = [
        SimpleNamespace(
            low=np.asarray([-1.0, -0.83, 0.0], dtype="float32"),
            high=np.asarray([1.0, 0.83, 1.0], dtype="float32"),
        )
    ]

    expert = [[0.0, 0.0, 0.5]]
    centered_policy_action = [[0.0, 0.0, 0.25]]
    high_policy_action = [[0.5, 0.415, 0.5]]

    assert controller._compose_residual_actions(expert, centered_policy_action) == expert
    composed = controller._compose_residual_actions(expert, high_policy_action)

    np.testing.assert_allclose(composed[0], [0.375, 0.31125, 0.6875], rtol=1e-6)

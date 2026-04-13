from __future__ import annotations

from cos435_citylearn.algorithms.sac.rewards import OfficialChallengeReward


def _observation(
    *,
    net: float = 2.0,
    carbon: float = 0.5,
    indoor: float = 25.0,
    set_point: float = 22.0,
    occupants: float = 1.0,
    outage: float = 0.0,
    soc: float = 0.5,
) -> dict[str, float]:
    return {
        "net_electricity_consumption": net,
        "carbon_intensity": carbon,
        "indoor_dry_bulb_temperature": indoor,
        "indoor_dry_bulb_temperature_set_point": set_point,
        "occupant_count": occupants,
        "power_outage": outage,
        "electrical_storage_soc": soc,
    }


def test_reward_reset_clears_ramping_state() -> None:
    reward = OfficialChallengeReward({"central_agent": True}, version="reward_v1")
    first = reward.calculate([_observation(net=1.0)])[0]
    second = reward.calculate([_observation(net=6.0)])[0]

    reward.reset()
    after_reset = reward.calculate([_observation(net=6.0)])[0]

    assert second < after_reset
    assert first <= 0.0
    assert after_reset <= 0.0


def test_reward_v2_penalizes_storage_smoothness_and_sign_flips() -> None:
    reward_v1 = OfficialChallengeReward({"central_agent": True}, version="reward_v1")
    reward_v2 = OfficialChallengeReward({"central_agent": True}, version="reward_v2")
    base_step = [_observation(soc=0.40)]
    flip_step = [_observation(soc=0.70)]
    sign_flip_step = [_observation(soc=0.35)]

    reward_v1.calculate(base_step)
    reward_v2.calculate(base_step)
    reward_v1.calculate(flip_step)
    reward_v2.calculate(flip_step)
    baseline_v1 = reward_v1.calculate(sign_flip_step)[0]
    penalized_v2 = reward_v2.calculate(sign_flip_step)[0]

    assert penalized_v2 < baseline_v1


def test_reward_broadcasts_for_decentralized_mode() -> None:
    reward = OfficialChallengeReward({"central_agent": False}, version="reward_v1")
    values = reward.calculate(
        [
            _observation(net=2.0, outage=1.0),
            _observation(net=1.0, outage=0.0),
        ]
    )

    assert len(values) == 2
    assert values[0] == values[1]


def test_main_comfort_term_excludes_outage_hours() -> None:
    reward = OfficialChallengeReward({"central_agent": True}, version="reward_v1")

    comfort_term = reward._comfort_excess(  # noqa: SLF001
        [_observation(indoor=28.0, set_point=22.0, outage=1.0)],
        outage_only=False,
    )
    outage_comfort_term = reward._comfort_excess(  # noqa: SLF001
        [_observation(indoor=28.0, set_point=22.0, outage=1.0)],
        outage_only=True,
    )

    assert comfort_term == 0.0
    assert outage_comfort_term > 0.0

from __future__ import annotations

import pytest

from cos435_citylearn.algorithms.ppo.schedules import LinearSchedule, parse_ent_coef


def test_linear_schedule_start_and_clamp() -> None:
    schedule = LinearSchedule(start=0.02, end=0.0, anneal_fraction=0.5)
    assert schedule.value_at(0.0) == pytest.approx(0.02)
    # midpoint of the anneal window -> halfway between start and end
    assert schedule.value_at(0.25) == pytest.approx(0.01)
    # end of anneal window -> exactly end
    assert schedule.value_at(0.5) == pytest.approx(0.0)
    # clamped past anneal window
    assert schedule.value_at(0.9) == pytest.approx(0.0)
    # clamped above 1.0
    assert schedule.value_at(5.0) == pytest.approx(0.0)


def test_linear_schedule_rejects_zero_anneal_fraction() -> None:
    with pytest.raises(ValueError, match="anneal_fraction"):
        LinearSchedule(start=0.02, end=0.0, anneal_fraction=0.0)


def test_parse_ent_coef_accepts_float() -> None:
    initial, schedule = parse_ent_coef(0.01)
    assert initial == pytest.approx(0.01)
    assert schedule is None


def test_parse_ent_coef_accepts_mapping() -> None:
    initial, schedule = parse_ent_coef(
        {"start": 0.02, "end": 0.0, "anneal_fraction": 0.5}
    )
    assert initial == pytest.approx(0.02)
    assert schedule is not None
    assert schedule.value_at(1.0) == pytest.approx(0.0)

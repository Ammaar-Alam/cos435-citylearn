from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LinearSchedule:
    start: float
    end: float
    anneal_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.anneal_fraction <= 1.0:
            raise ValueError(
                f"anneal_fraction must be in (0, 1]; got {self.anneal_fraction}"
            )

    def value_at(self, progress: float) -> float:
        # linearly decay from start -> end over [0, anneal_fraction], then hold at end
        p = max(0.0, min(1.0, float(progress)))
        if p >= self.anneal_fraction:
            return float(self.end)
        t = p / self.anneal_fraction
        return float(self.start + (self.end - self.start) * t)

    def as_mapping(self) -> dict[str, float]:
        return {
            "start": float(self.start),
            "end": float(self.end),
            "anneal_fraction": float(self.anneal_fraction),
        }


def parse_ent_coef(value: Any) -> tuple[float, LinearSchedule | None]:
    if isinstance(value, Mapping):
        schedule = LinearSchedule(
            start=float(value["start"]),
            end=float(value["end"]),
            anneal_fraction=float(value.get("anneal_fraction", 1.0)),
        )
        return schedule.value_at(0.0), schedule

    if value is None:
        return 0.0, None

    return float(value), None

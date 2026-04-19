# count-invariant shared context so a 3-building policy transfers to 6 buildings
from __future__ import annotations

from typing import Sequence

import numpy as np

SHARED_CONTEXT_V2_DIMENSION = 4


def _observation_value(
    observation: Sequence[float],
    names: Sequence[str],
    target_name: str,
    default: float = 0.0,
) -> float:
    try:
        index = names.index(target_name)
    except ValueError:
        return default

    return float(observation[index])


def build_shared_context_v2(
    observations: Sequence[Sequence[float]],
    observation_names: Sequence[Sequence[str]],
) -> np.ndarray:
    positive_loads: list[float] = []
    storage_socs: list[float] = []
    outages: list[float] = []

    for values, names in zip(observations, observation_names):
        net_load = _observation_value(values, names, "net_electricity_consumption")
        positive_loads.append(max(net_load, 0.0))
        storage_socs.append(_observation_value(values, names, "electrical_storage_soc"))
        outages.append(1.0 if _observation_value(values, names, "power_outage") > 0.0 else 0.0)

    positive_load_mean = float(np.mean(positive_loads, dtype="float32")) if positive_loads else 0.0
    mean_storage_soc = float(np.mean(storage_socs, dtype="float32")) if storage_socs else 0.0
    max_storage_soc = float(np.max(storage_socs)) if storage_socs else 0.0
    outage_fraction = float(np.mean(outages, dtype="float32")) if outages else 0.0

    return np.asarray(
        [positive_load_mean, mean_storage_soc, max_storage_soc, outage_fraction],
        dtype="float32",
    )

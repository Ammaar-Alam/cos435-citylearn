from __future__ import annotations

from typing import Sequence

import numpy as np

SHARED_CONTEXT_DIMENSION = 4


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


def build_shared_context(
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

    positive_load_sum = float(np.sum(positive_loads, dtype="float32"))
    positive_load_mean = float(np.mean(positive_loads, dtype="float32")) if positive_loads else 0.0
    mean_storage_soc = float(np.mean(storage_socs, dtype="float32")) if storage_socs else 0.0
    outage_fraction = float(np.mean(outages, dtype="float32")) if outages else 0.0

    return np.asarray(
        [positive_load_sum, positive_load_mean, mean_storage_soc, outage_fraction],
        dtype="float32",
    )


def augment_shared_observations(
    observations: Sequence[Sequence[float]],
    observation_names: Sequence[Sequence[str]],
) -> list[list[float]]:
    shared_context = build_shared_context(observations, observation_names)
    augmented: list[list[float]] = []

    for observation in observations:
        merged = np.concatenate(
            [np.asarray(observation, dtype="float32"), shared_context],
            dtype="float32",
        )
        augmented.append(merged.astype(float).tolist())

    return augmented

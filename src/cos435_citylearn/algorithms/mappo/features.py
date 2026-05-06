from __future__ import annotations

import numpy as np

CENTRALIZED_CRITIC_CONTEXT_VERSION = "centralized_critic_context_v1"


def centralized_critic_context_dimension(
    encoded_observation_dim: int,
    shared_context_dim: int,
) -> int:
    return int(encoded_observation_dim) * 4 + int(shared_context_dim)


def build_centralized_critic_context(
    encoded_observations: np.ndarray,
    shared_context: np.ndarray,
) -> np.ndarray:
    """Build a count-invariant district summary for MAPPO's centralized critic."""
    encoded = np.asarray(encoded_observations, dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[0] == 0:
        raise ValueError("centralized critic context requires a non-empty 2D observation array")

    shared = np.asarray(shared_context, dtype=np.float32).reshape(-1)
    return np.concatenate(
        [
            encoded.mean(axis=0),
            encoded.std(axis=0),
            encoded.min(axis=0),
            encoded.max(axis=0),
            shared,
        ],
        dtype=np.float32,
    )

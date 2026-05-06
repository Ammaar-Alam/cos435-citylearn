from __future__ import annotations

import numpy as np

from cos435_citylearn.algorithms.mappo.features import (
    build_centralized_critic_context,
    centralized_critic_context_dimension,
)


def test_centralized_critic_context_is_count_invariant() -> None:
    encoded_observations = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ],
        dtype=np.float32,
    )
    shared_context = np.asarray([0.1, 0.2], dtype=np.float32)

    context = build_centralized_critic_context(encoded_observations, shared_context)

    assert context.shape == (centralized_critic_context_dimension(3, 2),)
    expected = np.asarray(
        [
            2.0,
            4.0,
            6.0,
            np.std([1.0, 2.0, 3.0]),
            np.std([2.0, 4.0, 6.0]),
            np.std([3.0, 6.0, 9.0]),
            1.0,
            2.0,
            3.0,
            3.0,
            6.0,
            9.0,
            0.1,
            0.2,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(context, expected, atol=1e-6)


def test_centralized_critic_context_rejects_empty_observations() -> None:
    encoded_observations = np.zeros((0, 3), dtype=np.float32)
    shared_context = np.zeros(2, dtype=np.float32)

    try:
        build_centralized_critic_context(encoded_observations, shared_context)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty centralized critic context input should fail")

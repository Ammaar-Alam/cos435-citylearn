from __future__ import annotations

import numpy as np

from cos435_citylearn.algorithms.ppo.controllers import normalize_rollout_advantages


def test_global_normalize_advantages_enabled_zero_centers_unit_variance() -> None:
    # With the flag on, the helper must return a zero-mean, unit-variance
    # array -- this is the behavior the committed PPO sweep relied on.
    advantages = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = normalize_rollout_advantages(advantages, enabled=True)
    assert np.isclose(result.mean(), 0.0, atol=1e-5)
    assert np.isclose(result.std(), 1.0, atol=1e-4)


def test_global_normalize_advantages_disabled_preserves_raw_values() -> None:
    # Regression for the Codex P2 finding: previously the global pass ran
    # unconditionally, so normalize_advantage=False silently still produced
    # standardized advantages. The helper must now be a real off-switch.
    advantages = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = normalize_rollout_advantages(advantages, enabled=False)
    np.testing.assert_allclose(result, advantages, atol=1e-6)


def test_global_normalize_advantages_skips_singleton_to_avoid_nan() -> None:
    # A singleton advantage array has zero std, which would produce NaN if we
    # blindly divided. The helper short-circuits the same way the original
    # inline block did (size > 1 guard).
    advantages = np.array([3.14], dtype=np.float32)
    result = normalize_rollout_advantages(advantages, enabled=True)
    np.testing.assert_allclose(result, advantages, atol=1e-6)


def test_global_normalize_advantages_does_not_mutate_input() -> None:
    # The caller in finish_rollout reassigns the returned array back into
    # self.rollout_buffer.advantages; the helper must not mutate the input in
    # place, because other call sites (and tests) may share the reference.
    advantages = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    original = advantages.copy()
    _ = normalize_rollout_advantages(advantages, enabled=True)
    np.testing.assert_allclose(advantages, original, atol=1e-6)

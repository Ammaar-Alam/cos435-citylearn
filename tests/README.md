# test layout

The default test path is:

```bash
make test
make check
```

`make test` and `make check` intentionally ignore `tests/smoke/` because those
tests require the CityLearn benchmark package and downloaded dataset. Run the
benchmark smoke matrix with:

```bash
make install-benchmark
make download-citylearn
make smoke
```

## non-benchmark tests

Top-level `tests/test_*.py` files cover config loading, artifact stores, API
routers, dashboard build discovery, runner registries, run IDs, PPO math and
checkpoint contracts, SAC/TD3 checkpoint contracts, reward functions, and sweep
aggregation. These tests should stay runnable without raw benchmark data.

## benchmark smoke tests

The smoke tests are kept separate because each one protects a distinct
end-to-end path:

- `test_env_instantiation.py` covers environment/schema loading and action bounds.
- `test_random_policy_rollout.py` covers the minimal random-policy rollout.
- `test_rbc_rollout.py` covers the built-in RBC evaluation path.
- `test_rbc_ui_exports.py` covers RBC simulation export and official UI bundle shape.
- `test_ppo_rollout.py` covers centralized PPO smoke training and trace limits.
- `test_ppo_shared_rollout.py` covers shared-parameter PPO and cross-topology preflight behavior.
- `test_sac_rollout.py` covers centralized SAC, shared-parameter SAC, and trace-limit behavior.
- `test_td3_rollout.py` covers centralized TD3, shared-parameter TD3, and trace-limit behavior.

Shared benchmark-runtime and dataset guards live in `tests/smoke/helpers.py`.

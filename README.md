# COS435 CityLearn

Generalizable RL for neighborhood battery control in CityLearn.

This repository contains the shared benchmark foundation for the COS435 / ECE433 final project. The CityLearn environment, official 2023 dataset download flow, schema export, smoke path, and built-in RBC baseline are all wired here. PPO, SAC, and custom methods come later on top of this base.

## overview

The project is scoped around the 2023 CityLearn challenge and the comparison between:
- RBC
- centralized PPO
- centralized SAC
- later structured RL variants

Foundation in this repo now has:
- pinned Python 3.10 benchmark install
- official 2023 dataset download script
- canonical CityLearn loader and thin env adapters
- observation/action schema export
- random rollout smoke coverage
- built-in RBC evaluation path with metric export
- lightweight non-benchmark tests for the shared scaffold

## setup

### prerequisites

- python 3.10
- `uv` is recommended because the install script can use it to provision Python 3.10 automatically

If `uv` is not installed, make sure `python3.10` is already available on your machine before running the setup script.

### default install

This is the normal repo setup for development, checks, and scaffold work:

```bash
bash scripts/setup/install_env.sh
source .venv/bin/activate
make env-info
make check
```

### benchmark install

Use this for the real CityLearn environment and baseline path:

```bash
make install-benchmark
source .venv/bin/activate
make env-info
make download-citylearn
make env-schema
make smoke
make train-rbc
```

## benchmark target

The repo is being built around the 2023 challenge setup:

- `python 3.10`
- `CityLearn==2.1b12`
- `citylearn_challenge_2023_phase_2_local_evaluation` as the default public local-eval dataset
- official dataset DOI `10.18738/T8/SXFWTI`

The benchmark stack stays separate from the default install so the repo can still be bootstrapped quickly for read-only or scaffold-only work.

## common commands

```bash
make test
make check
make install-benchmark
make download-citylearn
make env-schema
make smoke
make train-rbc
make env-info
make repo-tree
```

## repository layout

```text
configs/    config contracts for env, train, eval, and splits
scripts/    setup helpers, schema export, smoke runners, and baseline entrypoints
src/        shared Python package, dataset/env utilities, and baseline code
tests/      scaffold checks plus benchmark smoke tests
data/       dataset manifests are tracked, raw benchmark files stay out of git
results/    generated manifests, metrics, and run artifacts stay out of git
```

## benchmark flow

1. `make install-benchmark`
2. `source .venv/bin/activate`
3. `make download-citylearn`
4. `make env-schema`
5. `make smoke`
6. `make train-rbc`

`make env-schema` writes:
- `results/manifests/environment_lock.json`
- `results/manifests/observation_action_schema.json`

`make train-rbc` writes:
- a run directory under `results/runs/`
- JSON metrics for the built-in RBC rollout
- a flat CSV row in `results/metrics/`

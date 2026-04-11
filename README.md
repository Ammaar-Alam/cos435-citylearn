# COS435 CityLearn

Generalizable RL for neighborhood battery control in CityLearn.

This repository contains the base scaffold for the COS435 / ECE433 final project. The benchmark-specific work is still staged for the next implementation pass, but the repo is already structured, installable, and ready for shared development.

## overview

The project is scoped around the 2023 CityLearn challenge and the comparison between:
- RBC
- centralized PPO
- centralized SAC
- later structured RL variants

Scaffold so far has:
- pinned project metadata for Python 3.10
- baseline config skeletons for RBC, PPO, SAC, and eval
- shared helper modules for repo paths/run ids
- install + check scripts
- (very) lightweight tests for scaffold structure

Scaffolding stuff to do next:
- dataset download logic
- a canonical CityLearn loader
- centralized or decentralized environment wrappers
- executable RBC, PPO, or SAC training/eval pipelines
- benchmark outputs or figures

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

### benchmark package install

The challenge-specific package pins are staged separately in `requirements/benchmark.txt`. Install them when the repo starts the actual CityLearn environment pass:

```bash
bash scripts/setup/install_env.sh requirements/benchmark.txt
source .venv/bin/activate
make env-info
```

That benchmark stack is staged here, but the full CityLearn runtime still needs to be validated:

## benchmark target

The repo is being built around the 2023 challenge setup:

- `python 3.10`
- `CityLearn==2.1b12`

The benchmark dependency pins are separated from the default install on purpose so the base repo stays easy to bootstrap while the full environment setup is still WIP.

## common commands

```bash
make test
make check
make env-info
make repo-tree
```

## repository layout

```text
configs/    config contracts for env, train, eval, and splits
scripts/    setup helpers and repo checks
src/        shared Python package and future project code
tests/      lightweight tests for scaffold behavior
data/       tracked roots only, downloaded data stays out of git
results/    tracked roots only, generated outputs stay out of git
```

## next implementation milestone

I've done just basic repo scaffolding so far, next is going to be *actually* setting up the environment:
- add dataset acquisition
- build the canonical CityLearn loader
- inspect env observation and action schema
- add the first real RBC smoke path

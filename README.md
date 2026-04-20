# COS435 CityLearn

Generalizable RL for neighborhood battery control in CityLearn.

This repository contains the shared benchmark foundation for the COS435 / ECE433 final project. The CityLearn environment, official 2023 dataset download flow, schema export, smoke path, built-in RBC baseline, and repo-local SAC runners are wired here. PPO remains a config contract for later work.

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
- centralized native-SAC baseline with checkpoint export
- parameter-shared decentralized SAC with shared district context
- repo-local reward ladder for SAC (`reward_v0` through `reward_v3`)
- default simulation-data export for completed evaluation runs
- local dashboard backend and frontend for launch, playback, and comparison
- lightweight non-benchmark tests for the shared scaffold

## setup

### prerequisites

- python 3.10
- node 20+ with `npm`
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
make train-sac
make train-sac-shared
```

## benchmark target

The repo is being built around the 2023 challenge setup:

- `python 3.10`
- `CityLearn==2.1b12`
- `citylearn_challenge_2023_phase_2_local_evaluation` as the default public local-eval dataset
- official dataset DOI `10.18738/T8/SXFWTI`

The benchmark stack stays separate from the default install so the repo can still be bootstrapped quickly for read-only or scaffold-only work.

Official 2023 leaderboard numbers, including the RBC baseline, the winning CHESCA scores, and the public vs private gaps, are documented in [docs/benchmark-reference.md](/Users/alam/GitHub/cos435-citylearn/docs/benchmark-reference.md:1).

## held-out eval and benchmark submission

The 2023 challenge trains on 3 buildings (phase-2 local_evaluation ≈ `public_dev`) and evaluates on all 6 buildings in phase 3. That 3 → 6 topology jump means a policy can only be evaluated on `phase_3_{1,2,3}` if its architecture is topology-invariant.

- `sac_shared_dtde_*` — shared-parameter per-building SAC. One actor-critic is called once per building. Eligible for phase-3 held-out evaluation and AICrowd submission.
- `ppo_shared_dtde_*` — shared-parameter per-building PPO. Mirrors the SAC-shared design (one actor + one critic, called once per building, GAE(λ) on-policy updates) with a count-invariant `shared_context_version=v2`. Also eligible for phase-3 held-out evaluation.
- `ppo_central_baseline` — centralized PPO (stable-baselines3). Fixed-topology reference number on `public_dev` only.
- `sac_central_*` — centralized SAC. Fixed-topology reference numbers on `public_dev` only.

The runners enforce this at eval time. Running `scripts/train/run_sac.py --artifact-id <central_checkpoint> --split phase_3_1` raises with a message naming the building-count mismatch; the same preflight runs for PPO via a sibling `topology.json` that each training run now writes alongside the model.

When evaluating a saved checkpoint with `--artifact-id`, you must also pass `--config <the training config for that checkpoint>` so the controller is rebuilt with the matching `control_mode`; mismatches are caught by `validate_checkpoint_runner_compatibility`.

## common commands

```bash
make test
make check
make install-benchmark
make download-citylearn
make env-schema
make smoke
make train-rbc
make train-sac
make train-sac-shared
make dashboard-install
make dashboard-build
make dashboard-backend
make dashboard-frontend
make ui
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
7. `make train-sac`

`make env-schema` writes:
- `results/manifests/environment_lock.json`
- `results/manifests/observation_action_schema.json`

`make train-rbc` writes:
- a run directory under `results/runs/`
- JSON metrics for the built-in RBC rollout
- a flat CSV row in `results/metrics/`
- a `SimulationData/<run_id>/` export under `results/ui_exports/`
- a playback payload under `results/ui_exports/playback/`

`make train-sac` and `make train-sac-shared` write:
- a run directory under `results/runs/`
- JSON metrics plus a flat CSV row in `results/metrics/`
- `checkpoint.pt` for later deterministic evaluation
- `training_curve.csv` with step-level optimization stats
- a `rollout_trace.json` preview trace
- a `SimulationData/<run_id>/` export and playback payload for completed full evaluations

You can validate the exported `SimulationData/...` tree against the official CityLearn UI upload contract with:

```bash
make check-ui-exports
```

## local dashboard

The dashboard is a repo-native localhost UI. It does not replace the benchmark path; it launches the same runner and reads the same run artifacts.

Install the frontend once:

```bash
make dashboard-install
```

The dashboard assumes the benchmark environment is already usable locally, so make sure you have already run:

```bash
make install-benchmark
source .venv/bin/activate
make download-citylearn
```

For the normal dev workflow:

```bash
source .venv/bin/activate
make ui
```

That starts:
- the FastAPI backend on `http://127.0.0.1:8001`
- the Vite dashboard on `http://127.0.0.1:5173`

If you want the script to also open a browser window when you run it:

```bash
make ui-open
```

If you want to run the two halves separately:

```bash
make dashboard-backend
make dashboard-frontend
```

To serve the built dashboard from the Python backend at `/dashboard`:

```bash
make dashboard-build
make dashboard-backend
```

The dashboard currently supports:
- launching the built-in RBC benchmark from the UI
- launching the centralized SAC baseline from the UI
- launching the centralized PPO baseline from the UI
- launching the shared SAC `reward_v2` runner from the UI
- launching the shared PPO `reward_v2` runner from the UI
- watching live preview payloads and worker logs while the benchmark job runs
- listing discovered runs from `results/runs/`
- inspecting one run with synchronized metrics, trace playback, and render media
- comparing multiple runs side by side
- importing playback payloads or other artifacts into a local registry
- inspecting imported playback payloads directly in the UI
- importing SAC or shared PPO checkpoints and evaluating them through a checkpoint-capable runner

The dashboard exposes the launchable runners (RBC, centralized SAC, centralized PPO, shared SAC `reward_v2`, and shared PPO `reward_v2`) rather than every config variant in `configs/train/`.

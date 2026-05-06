# MAPPO CTDE Experiment Log

This note records the current RL-centric pivot beyond the PPO/SAC/TD3 baselines.
It is intentionally separate from generated result CSVs because the MAPPO branch
has only smoke-level evidence so far.

## Rationale

The earlier PPO/SAC/TD3 sweep infrastructure gave a broad baseline matrix, but
the strongest portable phase-3 path needs a topology-invariant policy. MAPPO is
a rubric-safe RL pivot because it keeps decentralized, parameter-shared actors
that can run on 3-building and 6-building splits, while adding a centralized
critic during training for better district-level credit assignment. This is a
cleaner next step than a CHESCA-residual controller in the current repo because
there is no checked-in CHESCA implementation to wrap, and the course rubric
rewards a clear RL method with reproducible comparisons over a partially
external controller dependency.

## Implemented

- `configs/train/mappo/mappo_shared_ctde_reward_v2.yaml` defines the full MAPPO
  CTDE run.
- `configs/train/mappo/mappo_shared_ctde_smoke.yaml` and
  `configs/eval/mappo_smoke.yaml` define the local smoke path.
- `src/cos435_citylearn/algorithms/mappo/` contains the centralized critic
  context, checkpoint validation, and rollout buffer.
- `src/cos435_citylearn/baselines/mappo.py` wires train, checkpoint, eval, and
  metric export using the same artifact layout as shared PPO.
- `scripts/cluster/mappo_smoke.slurm` runs the one-cell cluster smoke and
  prefers `$ROOT_DIR/src` so separate Neuronic worktrees do not import stale
  editable-install source.
- `scripts/cluster/mappo_sweep.slurm` adds an 18-cell MAPPO-only matrix:
  `3 learning rates x 2 entropy coefficients x 3 seeds`.
- `scripts/cluster/aggregate_final_sweep.py --algos mappo` aggregates the MAPPO
  matrix without changing the existing PPO/SAC/TD3 final-sweep contract.

## Verified Local Evidence

Commands run from the MAPPO worktree:

```bash
PYTHONPATH="$PWD/src" "/Users/erikdyer/Obsidian Vault/Classes/S2026/COS435/Final Project/cos435-citylearn/.venv/bin/python" \
  -m pytest tests/test_aggregate_final_sweep.py tests/test_export_submission_results.py \
  tests/test_mappo_features.py tests/test_mappo_rollout_buffer.py \
  tests/test_mappo_controller.py tests/test_mappo_checkpoint.py tests/test_runner_registry.py -q
```

Result: `20 passed, 4 warnings`.

```bash
PYTHONPATH="$PWD/src" "/Users/erikdyer/Obsidian Vault/Classes/S2026/COS435/Final Project/cos435-citylearn/.venv/bin/python" \
  scripts/check/check_configs.py
```

Result: `checked 35 config files`.

```bash
bash -n scripts/cluster/run_final_cell.sh scripts/cluster/mappo_sweep.slurm scripts/cluster/submit_sweep.sh
```

Result: exit code `0`.

After the direct-`sbatch` and separate-worktree fixes:

```bash
PYTHONPATH="$PWD/src" "/Users/erikdyer/Obsidian Vault/Classes/S2026/COS435/Final Project/cos435-citylearn/.venv/bin/python" \
  -m pytest tests/test_cluster_scripts.py tests/test_aggregate_final_sweep.py -q
```

Result: `5 passed`.

```bash
PYTHONPATH="$PWD/src" "/Users/erikdyer/Obsidian Vault/Classes/S2026/COS435/Final Project/cos435-citylearn/.venv/bin/python" \
  -m ruff check tests/test_cluster_scripts.py
```

Result: `All checks passed!`.

```bash
bash -n scripts/cluster/run_final_cell.sh scripts/cluster/mappo_smoke.slurm scripts/cluster/mappo_sweep.slurm scripts/cluster/submit_sweep.sh
```

Result: exit code `0`.

Smoke metrics already generated locally:

- `public_dev` MAPPO smoke, 96 training steps and 64 eval steps:
  `average_score=0.6261109113693237`
- same checkpoint evaluated on `phase_3_1` with the capped smoke eval:
  `average_score=0.8594858646392822`

These are syntax and topology-portability checks only. They are not competitive
training evidence because the smoke config intentionally caps training and eval.

## Baseline Neuronic Matrix

Existing PPO/SAC/TD3 final-sweep job `2984866` completed all 81 array cells
with Slurm exit code `0:0`. Aggregation wrote 567 rows to:

```text
/n/fs/pvl-lidar/cache/ed1783/citylearn/sweeps/citylearn-final-hp-20260506-r1/summary.csv
```

Best mean `public_dev` configurations, using lower score as better:

| Algorithm | Selected config | public_dev mean | phase 2 mean | phase 3 mean |
| --- | --- | ---: | ---: | ---: |
| PPO | `lr=3e-3`, `ent_coef=0.0` | 0.770650 | 0.782011 | 0.816921 |
| SAC | `lr=1e-3`, `reward_scaling=5.0` | 0.553356 | 0.665839 | 0.743745 |
| TD3 | `lr=1e-3`, `exploration_noise=0.05` | 0.600946 | 0.730462 | 0.763056 |

SAC remains the strongest baseline from this matrix.

## MAPPO Neuronic Status

The current PPO/SAC/TD3 final sweep is complete. The MAPPO matrix is separate
and uses its own sweep id:

```bash
cd /u/ed1783/cos435-citylearn-mappo-work
JOB=mappo_sweep ROOT_DIR=/u/ed1783/cos435-citylearn-mappo-work \
  SWEEP_ID=citylearn-mappo-ctde-20260506-r1 \
  bash scripts/cluster/submit_sweep.sh
```

Submitted Slurm job: `2985130`.

Artifact root:

```text
/n/fs/pvl-lidar/cache/ed1783/citylearn/sweeps/citylearn-mappo-ctde-20260506-r1
```

Aggregate after completion:

```bash
python scripts/cluster/aggregate_final_sweep.py \
  --algos mappo \
  --sweep-root "$SWEEP_ROOT" \
  --out "$SWEEP_ROOT/summary.csv"
```

## What Worked

- The MAPPO checkpoint format supports a 3-building `public_dev` train artifact
  and a 6-building `phase_3_1` eval target.
- The centralized critic can consume count-invariant mean/std/min/max district
  summaries without changing the actor observation contract.
- The aggregation path can now preserve the original PPO/SAC/TD3 final sweep and
  separately aggregate a MAPPO-only matrix.

## What Remains Open

- Monitor MAPPO Slurm job `2985130` to completion and aggregate
  `citylearn-mappo-ctde-20260506-r1/summary.csv`.
- Compare MAPPO against PPO/SAC/TD3 on `public_dev`, released phase 2, and
  released phase 3 using normalized CSV rows.
- Refresh `submission/results/` and figures only after the MAPPO matrix produces
  complete metrics.

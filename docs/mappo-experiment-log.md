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
- `scripts/cluster/mappo_smoke.slurm` runs the one-cell cluster smoke.
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

Smoke metrics already generated locally:

- `public_dev` MAPPO smoke, 96 training steps and 64 eval steps:
  `average_score=0.6261109113693237`
- same checkpoint evaluated on `phase_3_1` with the capped smoke eval:
  `average_score=0.8594858646392822`

These are syntax and topology-portability checks only. They are not competitive
training evidence because the smoke config intentionally caps training and eval.

## Neuronic Status

The current PPO/SAC/TD3 final sweep should remain undisturbed. The MAPPO matrix
is separate and should use its own sweep id:

```bash
cd /u/$USER/cos435-citylearn
JOB=mappo_sweep SWEEP_ID=citylearn-mappo-ctde-YYYYMMDD-r1 bash scripts/cluster/submit_sweep.sh
```

Aggregate after completion:

```bash
python scripts/cluster/aggregate_final_sweep.py \
  --algos mappo \
  --sweep-root "$SWEEP_ROOT" \
  --out "$SWEEP_ROOT/summary.csv"
```

The last attempted SSH status check failed with
`Permission denied (keyboard-interactive)`, so no MAPPO Neuronic job was
submitted from this session.

## What Worked

- The MAPPO checkpoint format supports a 3-building `public_dev` train artifact
  and a 6-building `phase_3_1` eval target.
- The centralized critic can consume count-invariant mean/std/min/max district
  summaries without changing the actor observation contract.
- The aggregation path can now preserve the original PPO/SAC/TD3 final sweep and
  separately aggregate a MAPPO-only matrix.

## What Remains Open

- Run the MAPPO 18-cell matrix on Neuronic after SSH access is restored and
  after confirming it will not contend with the current `cos435-final` array.
- Compare MAPPO against PPO/SAC/TD3 on `public_dev`, released phase 2, and
  released phase 3 using normalized CSV rows.
- Refresh `submission/results/` and figures only after the MAPPO matrix produces
  complete metrics.

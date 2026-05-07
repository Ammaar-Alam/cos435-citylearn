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

The current PPO/SAC/TD3 final sweep is complete. The MAPPO matrix was run as a
separate additive sweep:

```bash
cd /u/ed1783/cos435-citylearn-mappo-work
JOB=mappo_sweep ROOT_DIR=/u/ed1783/cos435-citylearn-mappo-work \
  SWEEP_ID=citylearn-mappo-ctde-20260506-r1 \
  bash scripts/cluster/submit_sweep.sh
```

Submitted Slurm job: `2985130`. All 18 array cells completed with Slurm exit
code `0:0`. Aggregation wrote 126 rows to:

```text
/n/fs/pvl-lidar/cache/ed1783/citylearn/sweeps/citylearn-mappo-ctde-20260506-r1/summary.csv
```

Artifact root:

```text
/n/fs/pvl-lidar/cache/ed1783/citylearn/sweeps/citylearn-mappo-ctde-20260506-r1
```

Aggregation command:

```bash
python scripts/cluster/aggregate_final_sweep.py \
  --algos mappo \
  --sweep-root "$SWEEP_ROOT" \
  --out "$SWEEP_ROOT/summary.csv"
```

Best MAPPO mean `public_dev` configuration:

| Algorithm | Selected config | public_dev mean | phase 2 mean | phase 3 mean |
| --- | --- | ---: | ---: | ---: |
| MAPPO | `lr=1e-3`, `ent_coef=0.0` | 0.922230 | 0.983547 | 0.997916 |

MAPPO did not beat any PPO/SAC/TD3 baseline in this matrix. Its best released
average was `0.960650` for `lr=1e-3`, `ent_coef=0.01`, still far worse than
the best SAC released average.

## Final Matrix Decision

The combined 33-config final hyperparameter summary is tracked in
`submission/results/final_hp_sweep_summary.csv`. Lower scores are better.

| Selection rule | Algorithm | Config | public_dev mean | phase 2 mean | phase 3 mean | released avg |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Best `public_dev` | SAC | `lr=1e-3`, `reward_scaling=5.0` | 0.553356 | 0.665839 | 0.743745 | 0.704792 |
| Best phase 2 | SAC | `lr=5e-4`, `reward_scaling=10.0` | 0.560255 | 0.663757 | 0.753156 | 0.708457 |
| Best phase 3 | SAC | `lr=5e-4`, `reward_scaling=5.0` | 0.566970 | 0.666234 | 0.737590 | 0.701912 |
| Best released average | SAC | `lr=5e-4`, `reward_scaling=5.0` | 0.566970 | 0.666234 | 0.737590 | 0.701912 |

Conclusion: the MAPPO pivot was rubric-safe and functional, but not useful for
leaderboard performance. The strongest reportable controller remains shared SAC.
Use `lr=1e-3`, `reward_scaling=5.0` as the primary configuration because it was
selected on `public_dev`, the tuning split. Report `lr=5e-4`,
`reward_scaling=5.0` as a post-hoc released-diagnostic configuration: it is the
best phase-3 and released-average SAC setting, but it should not be presented as
the pre-registered model-selection result.

## What Worked

- The MAPPO checkpoint format supports a 3-building `public_dev` train artifact
  and a 6-building `phase_3_1` eval target.
- The centralized critic can consume count-invariant mean/std/min/max district
  summaries without changing the actor observation contract.
- The aggregation path can now preserve the original PPO/SAC/TD3 final sweep and
  separately aggregate a MAPPO-only matrix.
- The MAPPO branch produced a complete seed/hyperparameter matrix quickly on
  Neuronic without interfering with the existing baseline sweep artifacts.

## What Did Not Work

- MAPPO underperformed badly after real training. Its best `public_dev` score
  was `0.922230`, compared with SAC at `0.553356` and TD3 at `0.600946`.
- The centralized critic did not translate into better cross-topology transfer
  under the current short shared-actor training budget.
- The separate MAPPO sweep initially exposed two workflow bugs: direct `sbatch`
  did not export `SWEEP_ROOT`, and the remote shared virtualenv imported the
  base checkout until cluster entrypoints explicitly preferred `$ROOT_DIR/src`.

## What Remains Open

- If figures need to show the final MAPPO matrix, copy the remote normalized
  MAPPO metric CSVs into local ignored `results/metrics/` and regenerate the
  report figures from the exporter path.

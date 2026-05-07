# Residual SAC Experiment Note

This note records the current hybrid expert-plus-RL stretch experiment for the
CityLearn 2023 Track-3 project. It is a handoff and provenance note, not a
final result claim.

## Motivation

The final-project rubric rewards a concerted effort to beat competition
baselines and other competitors. The existing project result is already
rubric-safe as an RL competition study: SAC and PPO variants beat RBC on
released evaluation splits, while CHESCA remains an official leaderboard
reference that this repo should not claim to have beaten without comparable
evidence.

Residual SAC is a stretch attempt to improve the strongest RL story with a
hybrid controller:

- an expert controller proposes a domain-informed action;
- a shared SAC policy learns a bounded residual correction;
- the shared policy remains building-count portable in the same spirit as the
  shared-DTDE SAC baseline.

## Implemented Method

The branch `feat/rl-hybrid-residual-controller` adds:

- `ExpertActionPolicy` with `basic_rbc` and `adaptive_storage_v1` experts;
- `ResidualSharedSACController`, which composes expert actions with centered
  SAC residual actions;
- residual SAC checkpoint save/load and topology-compatible evaluation;
- residual SAC configs under `configs/train/sac/`;
- `scripts/eval/run_expert_policy.py` for expert-only scoring;
- `scripts/cluster/residual_sac_sweep.slurm` for a 27-cell sweep, with
  `RESIDUAL_EXPERT_POLICY` support for a BasicRBC follow-up;
- `scripts/cluster/aggregate_hp_sweep.py` for complete-cell aggregation.

## Current Expert-Only Baselines

Lower score is better. These numbers are local CSV outputs from full expert
evaluation runs and should be treated as diagnostics for the residual starting
point.

| expert policy | public_dev | released avg over phase_2 and phase_3 |
| --- | ---: | ---: |
| `adaptive_storage_v1` | 1.072494 | 1.127151 |
| `basic_rbc` | 1.022619 | 1.100401 |

Interpretation: the adaptive expert is not a strong standalone controller.
BasicRBC is better on every measured split. A residual policy can still improve
over its expert, but beating CHESCA would require a very large improvement from
this starting point.

## Neuronic Sweep Provenance

The first residual submission failed before training:

- job: `2985311`
- sweep root:
  `/n/fs/pvl-lidar/cache/ed1783/citylearn/sweeps/citylearn-residual-sac-20260507-r1`
- failure:
  `TypeError: run_sac() got an unexpected keyword argument 'expert_policy_override'`
- cause: reused editable environment imported the main checkout instead of the
  residual worktree sources.
- fix: `scripts/cluster/run_final_cell.sh` now prepends
  `$ROOT_DIR/src` to `PYTHONPATH`.

The active clean submission is:

- job: `2985338`
- sweep root:
  `/n/fs/pvl-lidar/cache/ed1783/citylearn/sweeps/citylearn-residual-sac-20260507-r2`
- matrix: `3` learning rates x `3` residual scales x `3` seeds = `27` cells
- config:
  `configs/train/sac/sac_shared_residual_adaptive_reward_v2.yaml`
- eval config: `configs/eval/official_released.yaml`

Last verified live state before SSH reauthentication was required:

- time: 2026-05-07 around 00:56 EDT
- all `27` Slurm array tasks were still running;
- Python workers were active and CPU-bound;
- no non-empty `train.json`, `eval_*.json`, or summary CSV artifacts existed
  yet.

## Aggregation Command

After Neuronic auth is restored and the array finishes, aggregate the sweep on
the remote worktree:

```bash
cd /u/ed1783/cos435-citylearn-residual
SWEEP_ROOT=/n/fs/pvl-lidar/cache/ed1783/citylearn/sweeps/citylearn-residual-sac-20260507-r2

python scripts/cluster/aggregate_hp_sweep.py \
  --sweep-root "$SWEEP_ROOT" \
  --out "$SWEEP_ROOT/summary.csv"
```

The aggregator should fail unless all expected residual cells and splits are
present. Do not use `--allow-missing` for final claims.

## Go/No-Go Rule

Use the released average over phase-2 online-eval and phase-3 splits for the
first decision:

- `> 0.80`: stop chasing CHESCA with this method; document as a negative stretch
  result.
- `0.70 - 0.80`: do not claim CHESCA competitiveness; consider only one small
  follow-up, likely switching the expert base to `basic_rbc` with
  `RESIDUAL_EXPERT_POLICY=basic_rbc`.
- `< 0.70`: worth targeted tuning because it is competitive with the strongest
  existing RL results.
- `< 0.60`: worth a serious CHESCA-focused follow-up.
- `<= 0.565`: CHESCA-competitive by the official private reference threshold,
  subject to the repo's caveat that released local/eval splits are not the same
  evaluator as the original leaderboard.

Until `summary.csv` exists and has complete coverage, residual SAC should be
reported as an in-progress stretch experiment rather than a final result.

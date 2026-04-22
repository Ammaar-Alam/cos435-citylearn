# Evaluation Splits and Dataset Policy

This repo uses more than one CityLearn 2023 dataset family, and the distinction
matters for every project claim.

## split families

| split config | dataset name | role in this repo | tuning allowed |
| --- | --- | --- | --- |
| `public_dev` | `citylearn_challenge_2023_phase_2_local_evaluation` | default local development and baseline tuning split | yes |
| `phase_2_online_eval_1` | `citylearn_challenge_2023_phase_2_online_evaluation_1` | released phase-2 online-evaluation dataset 1 | no |
| `phase_2_online_eval_2` | `citylearn_challenge_2023_phase_2_online_evaluation_2` | released phase-2 online-evaluation dataset 2 | no |
| `phase_2_online_eval_3` | `citylearn_challenge_2023_phase_2_online_evaluation_3` | released phase-2 online-evaluation dataset 3 | no |
| `phase_3_1` | `citylearn_challenge_2023_phase_3_1` | released phase-3 six-building dataset 1 | no |
| `phase_3_2` | `citylearn_challenge_2023_phase_3_2` | released phase-3 six-building dataset 2 | no |
| `phase_3_3` | `citylearn_challenge_2023_phase_3_3` | released phase-3 six-building dataset 3 | no |

The public local split is for development. The released phase-2 online-eval and
phase-3 datasets should be treated as evaluation-only.

## why `public_dev` is not enough

`public_dev` points at the released local-evaluation dataset and is the correct
place to tune baselines and reward variants inside this repo.

It is not the same thing as the leaderboard evaluator used during the 2023
competition. The official challenge documentation explains that:

- local evaluation used a released three-building dataset
- the phase-2 public leaderboard used evaluator-side datasets with different
  private outage seeds
- the final private leaderboard used six buildings and different private seeds

Because of that, a strong `public_dev` score can show that a method is promising
but it is not, by itself, enough to claim that the repo matched or beat the
official leaderboard winner.

## what can be reproduced locally now

The official 2023 dataset DOI used by this repo publishes the released
post-competition dataset family, including:

- `citylearn_challenge_2023_phase_2_local_evaluation`
- `citylearn_challenge_2023_phase_2_online_evaluation_1`
- `citylearn_challenge_2023_phase_2_online_evaluation_2`
- `citylearn_challenge_2023_phase_2_online_evaluation_3`
- `citylearn_challenge_2023_phase_3_1`
- `citylearn_challenge_2023_phase_3_2`
- `citylearn_challenge_2023_phase_3_3`

That means the repo can now perform a released-dataset checkpoint evaluation
pass that is much closer to the official challenge setting than `public_dev`.

This is the right way to test a saved final SAC checkpoint after local tuning.

## checkpoint portability by controller family

There is one important architectural caveat in this repo:

- centralized SAC checkpoints trained on the three-building `public_dev` split
  are portable to the released `phase_2_online_eval_*` datasets because those
  splits keep the same building count
- centralized SAC checkpoints are not portable to the released six-building
  `phase_3_*` datasets because the centralized observation and action schema
  changes with building count
- shared DTDE SAC checkpoints are designed to reuse one policy across buildings
  and can be evaluated on the six-building `phase_3_*` splits as long as the
  per-building schema stays consistent

## expected workflow

1. tune on `public_dev`
2. lock the chosen checkpoint
3. evaluate that checkpoint on all `phase_2_online_eval_*` splits
4. if the controller is building-count portable, evaluate that checkpoint on all
   `phase_3_*` splits
5. report those numbers separately from the local-dev comparisons

Do not tune on the released online-eval or phase-3 datasets if the goal is a
clean generalization claim.

## downloading datasets

The normal repo bootstrap only downloads the default local-evaluation dataset:

```bash
make download-citylearn
```

To download the full released 2023 dataset family, run:

```bash
make download-citylearn-all
```

The tracked metadata file
[`data/manifests/citylearn_2023_manifest.json`](../data/manifests/citylearn_2023_manifest.json)
records the released dataset inventory and checksums.

## checkpoint evaluation

The repo provides an evaluation-only SAC path for released official datasets:

```bash
COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(pwd)/.cache/matplotlib" .venv/bin/python \
  scripts/eval/run_sac_checkpoint.py \
  --config configs/train/sac/sac_central_reward_v1.yaml \
  --eval-config configs/eval/official_released.yaml \
  --checkpoint-path results/runs/sac__central_reward_v1__public_dev__seed2__20260420_133020/checkpoint.pt \
  --split phase_2_online_eval_1 \
  --seed 2
```

This path:

- loads an existing checkpoint
- skips training
- evaluates deterministically on the requested split
- writes a new metric row and run manifest under `results/`
- disables simulation-data export and render capture by default to keep evals fast

## what should be committed

Commit:

- split config files under `configs/splits/`
- downloader and manifest code
- tracked metadata under `data/manifests/`
- scripts and docs that define the evaluation workflow
- clean tracked summaries derived from raw metrics

Do not commit:

- raw datasets under `data/external/`
- raw run artifacts under `results/runs/`, `results/metrics/`, or `results/ui_exports/`
- checkpoints, traces, or Colab/Drive-specific operator notes

The repo source of truth for dataset reproducibility is the DOI plus the tracked
manifest metadata, not the raw dataset files inside git.

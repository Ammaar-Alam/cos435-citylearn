# Current Results Snapshot

These files are the clean tracked summary of the raw outputs under `results/`.

## Files in this directory

- `local_main_results.csv` — tracked method-level summary rows
- `sac_ablation_summary.csv` — SAC-only variant comparison with seed-aware means and CIs
- `sac_seed_inventory.csv` — per-seed SAC run inventory for the local phase-2 batch
- `released_eval_main_results.csv` — released official-eval family summaries
- `released_eval_seed_inventory.csv` — per-seed released-eval checkpoint inventory
- `official_benchmark_reference.csv` — published CityLearn 2023 reference numbers

## Local public_dev snapshot

- local RBC baseline: `1.022619`
- PPO baseline artifact: missing locally
- Centralized SAC baseline: mean `0.553964`, std `0.009170`, 95% CI `0.011386`, seeds `5`
- Centralized SAC reward_v1: mean `0.527309`, std `0.011312`, 95% CI `0.014046`, seeds `5`
- Centralized SAC reward_v2: mean `0.535651`, std `0.015883`, 95% CI `0.019722`, seeds `5`
- Shared DTDE SAC reward_v2: mean `0.568877`, std `0.017638`, 95% CI `0.043814`, seeds `3`

Lower is better.

## Local tuning headline

The best current mean is `central_reward_v1` at `0.527309` across `5` seeds.

The strongest improved SAC result with a full 5-seed comparison is
`central_reward_v1` at `0.527309`.
That is `0.495310` lower than the local RBC baseline, a
`48.44%` improvement on the local phase-2 evaluation dataset.

## Released official-eval snapshot

- Centralized SAC reward_v2 on `released_phase_2_online_eval`: mean `0.652703`, std `0.042913`, 95% CI `0.021717`, eval jobs `15`, seeds `5`
- Centralized SAC baseline on `released_phase_2_online_eval`: mean `0.662064`, std `0.036729`, 95% CI `0.018588`, eval jobs `15`, seeds `5`
- Centralized SAC reward_v1 on `released_phase_2_online_eval`: mean `0.667699`, std `0.045280`, 95% CI `0.022915`, eval jobs `15`, seeds `5`
- Shared DTDE SAC reward_v2 on `released_phase_2_online_eval`: mean `0.676669`, std `0.047188`, 95% CI `0.036272`, eval jobs `9`, seeds `3`
- The shared DTDE checkpoint family also completed the released `phase_3_*` six-building sweep at `0.774245` across `9` eval jobs.

Lower is better.

## Current headline

The released phase-2 winner among saved checkpoints is `central_reward_v2` at `0.652703` across `15` eval jobs.

## What the full SAC evidence says

- every measured SAC variant beats the local RBC baseline by a large margin
- the central SAC baseline, central `reward_v1`, and central `reward_v2` now all have claim-quality 5-seed local comparisons
- `central_reward_v1` currently beats `central_reward_v2` on mean total score (`0.527309` vs `0.535651`)
- `central_reward_v2` is the best saved fixed-topology checkpoint family on the released phase-2 online-eval datasets
- shared / decentralized `reward_v2` did not beat the released phase-2 central winner
- `phase_2_online_eval_1`: central `reward_v2` `0.690962` vs shared `reward_v2` `0.714750`
- `phase_2_online_eval_2`: central `reward_v2` `0.670103` vs shared `reward_v2` `0.700031`
- `phase_2_online_eval_3`: central `reward_v2` `0.597044` vs shared `reward_v2` `0.615225`
- none of the saved checkpoints currently beat the published CHESCA references

## Important caveats

- these numbers are local phase-2 evaluation numbers, not official leaderboard results
- some local SAC means are numerically below the published CHESCA references, but that is **not** enough to claim a true leaderboard win
- there is still no saved PPO artifact in this checkout, so PPO vs SAC is not yet empirical
- the released phase-2 online-eval datasets are much closer to the official evaluator-side setting than `public_dev`, but they are still reported separately from the local tuning split
- centralized checkpoints are not portable to the released six-building `phase_3_*` datasets, so the current `phase_3` evidence is shared-DTDE-only

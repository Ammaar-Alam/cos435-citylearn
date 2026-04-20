# Current Local Results Snapshot

These files are the clean tracked summary of the raw outputs under `results/`.

## Files in this directory

- `local_main_results.csv` — tracked method-level summary rows
- `sac_ablation_summary.csv` — SAC-only variant comparison with seed-aware means and CIs
- `sac_seed_inventory.csv` — per-seed SAC run inventory for the local phase-2 batch
- `official_benchmark_reference.csv` — published CityLearn 2023 reference numbers

## What is currently measured

- local RBC baseline: `1.022619`
- PPO baseline artifact: missing locally
- Centralized SAC baseline: mean `0.553964`, std `0.009170`, 95% CI `0.011386`, seeds `5`
- Centralized SAC reward_v1: mean `0.525948`, std `0.013010`, 95% CI `0.032320`, seeds `3`
- Centralized SAC reward_v2: mean `0.535651`, std `0.015883`, 95% CI `0.019722`, seeds `5`
- Shared DTDE SAC reward_v2: mean `0.568877`, std `0.017638`, 95% CI `0.043814`, seeds `3`

Lower is better.

## Current headline

The best pilot mean right now is `central_reward_v1` at `0.525948`,
but that result only has `3` seeds.

The strongest improved SAC result with a full 5-seed comparison is
`central_reward_v2` at `0.535651`.
That is `0.486967` lower than the local RBC baseline, a
`47.62%` improvement on the local phase-2 evaluation dataset.

## What the overnight SAC batch says

- every measured SAC variant beats the local RBC baseline by a large margin
- the central SAC baseline and central `reward_v2` now have claim-quality 5-seed local comparisons
- central `reward_v1` has the best current pilot mean, but only on 3 seeds
- shared / decentralized `reward_v2` did not beat the best centralized SAC result in this batch

## Important caveats

- these numbers are local phase-2 evaluation numbers, not official leaderboard results
- some local SAC means are numerically below the published CHESCA references, but that is **not** enough to claim a true leaderboard win
- there is still no saved PPO artifact in this checkout, so PPO vs SAC is not yet empirical
- held-out evaluation is still missing, so the final paper claim is not complete yet

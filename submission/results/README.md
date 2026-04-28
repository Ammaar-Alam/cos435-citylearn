# Current Results Snapshot

These files are the clean tracked summary of the raw outputs under `results/`. Raw run directories, checkpoints, rollout traces, and downloaded Drive bundles remain outside git.

## Files in this directory

- `local_main_results.csv` - tracked method-level summary rows for local `public_dev`
- `method_comparison.csv` - compact report table for RBC, PPO, and SAC local comparisons
- `cross_split_scores.csv` - compact report table across `public_dev`, released phase-2, and released phase-3 splits
- `sac_ablation_summary.csv` - SAC-only variant comparison with seed-aware means and CIs
- `sac_seed_inventory.csv` - per-seed SAC run inventory for the local phase-2 batch
- `released_eval_main_results.csv` - released official-eval family summaries
- `released_eval_seed_inventory.csv` - per-run released-eval inventory for RBC, PPO, and SAC where artifacts exist
- `ppo_shared_sweep_summary.csv` - per-learning-rate shared-PPO sweep summary rows
- `ppo_shared_sweep_inventory.csv` - per-run shared-PPO sweep inventory with KPI columns
- `official_benchmark_reference.csv` - published CityLearn 2023 reference numbers
- `figure_manifest.csv` - tracked report figure inventory

## Local public_dev snapshot

Lower is better.

- Local RBC baseline: mean `1.022619`, std `0.000000`, 95% CI `0.000000`, seeds `1`
- Centralized PPO baseline: mean `0.882911`, std `0.011102`, 95% CI `0.011651`, seeds `6`
- Shared DTDE PPO reward_v2: mean `0.776976`, std `0.036542`, 95% CI `0.026141`, seeds `10`
- Centralized SAC reward_v1: mean `0.527309`, std `0.011312`, 95% CI `0.014046`, seeds `5`
- Centralized SAC reward_v2: mean `0.535651`, std `0.015883`, 95% CI `0.019722`, seeds `5`
- Centralized SAC baseline: mean `0.553964`, std `0.009170`, 95% CI `0.011386`, seeds `5`
- Shared DTDE SAC reward_v2: mean `0.568877`, std `0.017638`, 95% CI `0.043814`, seeds `3`

## PPO shared sweep detail

Per-learning-rate breakdown behind the shared-PPO summary row above:

- Shared DTDE PPO reward_v2 lr=`1e-4` on `public_dev`: mean `0.795652`, std `0.038388`, 95% CI `0.027461`, seeds `10`
- Shared DTDE PPO reward_v2 lr=`3e-4` on `public_dev`: mean `0.783261`, std `0.037731`, 95% CI `0.026991`, seeds `10`
- Shared DTDE PPO reward_v2 lr=`1e-4` across released `phase_3_*`: mean `0.847284`, std `0.031378`, 95% CI `0.011229`, eval jobs `30`
- Shared DTDE PPO reward_v2 lr=`3e-4` across released `phase_3_*`: mean `0.847783`, std `0.032190`, 95% CI `0.011519`, eval jobs `30`

The best shared-PPO local sweep setting is lr=`3e-4` at `0.783261` on `public_dev`.

## Local tuning headline

The best local mean is `Centralized SAC reward_v1` at `0.527309`. That is `0.495310` lower than the local RBC baseline, a `48.44%` improvement on the local phase-2 evaluation dataset.

## Released official-eval snapshot

Lower is better.

- RBC baseline on `released_phase_2_online_eval`: mean `1.087092`, std `0.020621`, 95% CI `0.051225`, eval jobs `3`, seeds `1`
- RBC baseline on `released_phase_3`: mean `1.113710`, std `0.012142`, 95% CI `0.030161`, eval jobs `3`, seeds `1`
- Centralized PPO baseline on `released_phase_2_online_eval`: mean `0.872718`, std `0.006820`, 95% CI `0.016942`, eval jobs `3`, seeds `1`
- Shared DTDE PPO reward_v2 on `released_phase_2_online_eval`: mean `0.793176`, std `0.054182`, 95% CI `0.019389`, eval jobs `30`, seeds `10`
- Shared DTDE PPO reward_v2 on `released_phase_3`: mean `0.843221`, std `0.030263`, 95% CI `0.010829`, eval jobs `30`, seeds `10`
- Centralized SAC reward_v2 on `released_phase_2_online_eval`: mean `0.652703`, std `0.042913`, 95% CI `0.021717`, eval jobs `15`, seeds `5`
- Centralized SAC baseline on `released_phase_2_online_eval`: mean `0.662064`, std `0.036729`, 95% CI `0.018588`, eval jobs `15`, seeds `5`
- Centralized SAC reward_v1 on `released_phase_2_online_eval`: mean `0.667699`, std `0.045280`, 95% CI `0.022915`, eval jobs `15`, seeds `5`
- Shared DTDE SAC reward_v2 on `released_phase_2_online_eval`: mean `0.676669`, std `0.047188`, 95% CI `0.036272`, eval jobs `9`, seeds `3`
- Shared DTDE SAC reward_v2 on `released_phase_3`: mean `0.774245`, std `0.012061`, 95% CI `0.009271`, eval jobs `9`, seeds `3`

## Current headline

The released phase-2 winner among saved checkpoints is `Centralized SAC reward_v2` at `0.652703` across `15` eval jobs.

## Report figures

Tracked PNG copies live under `submission/figures/`:

- `cross_split_comparison.png`
- `cross_split_kpi_breakdown.png`
- `generalization_gap.png`
- `kpi_breakdown.png`
- `method_comparison.png`
- `per_split_scores.png`
- `ppo_training_curve.png`

## Important caveats

- `public_dev` is the local public evaluation dataset, not the hidden official leaderboard split.
- Released phase-2 and phase-3 datasets are closer to the post-competition evaluator setting, but they are still reported separately from leaderboard references.
- Published CHESCA references remain benchmark context, not a direct apples-to-apples threshold for local `public_dev` figures.
- Centralized policies are not portable to released six-building phase-3 splits without architecture changes; shared DTDE policies are the portable comparison there.

# Current Local Results Snapshot

These files are the clean tracked summary of the raw outputs under `results/`.

## What is currently measured

- local RBC baseline: `1.022619`
- SAC central baseline: `0.544145`
- SAC central reward_v1: `0.529279`
- SAC central reward_v2: `0.520719`
- SAC shared dtde reward_v2: `0.584087`
- PPO baseline artifact: missing locally

Lower is better.

## Current headline

The best local SAC run is `central_reward_v2` at `0.520719`.
That is `0.501900` lower than the local RBC baseline, a
`49.08%` improvement on the local phase-2 evaluation dataset.

## What the current SAC ladder suggests

- reward shaping helped centralized SAC
- `reward_v2` beat `reward_v1`
- the current shared/decentralized `reward_v2` run is worse than the best centralized run
- these are still single-seed measurements, so they are not claim-quality yet

## Important caveats

- these numbers are local phase-2 evaluation numbers, not official leaderboard results
- there is still no saved PPO artifact in this checkout, so PPO vs SAC is not yet empirical
- all SAC rows here are from `seed 0` only
- final claims still need multi-seed and held-out evaluation

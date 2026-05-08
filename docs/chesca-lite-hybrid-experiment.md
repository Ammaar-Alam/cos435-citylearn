# CHESCA-Lite Hybrid Controller

## Motivation

The CityLearn 2023 winner, CHESCA, was not a pure deep-RL policy. It combined
comfort-preserving local controllers, forecast-aware storage decisions, and
community-level smoothing. This branch adds a lightweight deterministic variant
of that idea so we can test whether a competition-inspired hybrid baseline closes
the gap left by the MAPPO and residual-SAC negative results.

## Implementation

- Controller: `cos435_citylearn.algorithms.heuristics.ChescaLiteController`
- Config: `configs/train/rbc/chesca_lite.yaml`
- Runner path: `run_rbc`, with split/seed overrides added to
  `scripts/train/run_rbc.py`

The controller uses the central-agent observation/action surface:

- per-building `dhw_storage`
- per-building `electrical_storage`
- per-building `cooling_device`

It applies:

- comfort-first cooling with reduced overcooling,
- DHW storage rules based on SOC, demand, and hour,
- electrical storage rules based on load, price, carbon, solar, and hour,
- action smoothing to reduce ramping.
- explicit controller kwargs for local tuning of cooling scale, smoothing, and
  storage charge/discharge strength.

## Local Pilot Results

Lower is better. These are local reproducible CityLearn evaluations, not hidden
official leaderboard submissions.

| split | CHESCA-lite score |
| --- | ---: |
| public_dev | 0.618049 |
| phase_2_online_eval_1 | 0.680990 |
| phase_2_online_eval_2 | 0.677772 |
| phase_2_online_eval_3 | 0.594212 |
| phase_3_1 | 0.671487 |
| phase_3_2 | 0.664418 |
| phase_3_3 | 0.675197 |

Summary:

- released average: `0.660679`
- phase 2 released average: `0.650991`
- phase 3 released average: `0.670368`

Reference points from the tracked result tables:

- local RBC public_dev: `1.022619`
- shared DTDE SAC released phase 3 mean: `0.774245`
- official CHESCA private leaderboard reference: `0.565`

## Interpretation

This is a useful hybrid baseline and beats the previous shared-SAC phase-3 mean,
but it does not beat CHESCA. The controller substantially reduces normal comfort
violations and carbon relative to BasicRBC. A small local parameter screen found
that lowering the cooling action smoothing cap from `0.18` to `0.12`, then
lowering the cooling action scale to `0.6`, improved the released average from
`0.700245` to `0.660679`. The remaining gap is mostly outage resilience and
grid-smoothness metrics, especially thermal resilience, unserved energy,
ramping, and load factor.

One rejected screen bypassed storage action smoothing during outages so the
battery could jump immediately to the outage-discharge action. That reduced one
public-dev unserved-energy term, but worsened the representative released-screen
aggregate through thermal-resilience regressions, so the final controller keeps
storage actions ramp-limited during outages.

The score is strong enough to keep as a documented CHESCA-inspired stretch
result. It is not strong enough to claim that we beat the official winner.

## Reproduction Notes

The final local pilot used exports disabled for speed and wrote temporary
artifacts to `/private/tmp/chesca-lite-pilot/results_all_scale_0p6_delta_0p12`.

Representative command shape:

```bash
PYTHONPATH=src python scripts/train/run_rbc.py \
  --config configs/train/rbc/chesca_lite.yaml \
  --eval-config configs/eval/default.yaml \
  --split public_dev
```

For fast local comparisons, use an eval config with `export_simulation_data:
false` and `capture_render_frames: false`.

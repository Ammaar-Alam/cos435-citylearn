# config layout

These files are the shared config surface for the implemented benchmark paths.

- `env/` holds benchmark and path-level config
- `train/` holds method configs for RBC, PPO, SAC, and TD3
- `eval/` holds evaluation output settings
- `splits/` holds split names and intent, not downloaded data

The files here are live inputs to the repo runners:

- `train/rbc/rbc_builtin.yaml` drives the built-in RBC path
- `train/ppo/*.yaml` drives the centralized and shared-parameter PPO runners
- `train/ppo/*_smoke.yaml` are intentionally tiny configs for test coverage only
- `train/sac/*.yaml` drives the centralized and shared SAC runners
- `train/sac/*_smoke.yaml` are intentionally tiny configs for test coverage only
- `train/td3/*.yaml` drives the centralized SB3 TD3 runner and shared DTDE TD3 runner
- `train/td3/*_smoke.yaml` are intentionally tiny configs for test coverage only
- `eval/default.yaml` is the normal full-eval path
- `eval/official_released.yaml` is the released phase-2 and phase-3 evaluation path
- `eval/sac_smoke.yaml` disables simulation export and caps evaluation steps for fast PPO/SAC/TD3 smoke runs

RBC, PPO, SAC, and TD3 configs are live runner inputs. They are launchable through
the scripts under `scripts/train/`, the Make targets in the repo root, and the
dashboard runner registry for the exposed variants. Centralized PPO/SAC/TD3
configs are fixed-topology references for `public_dev`; shared PPO/SAC/TD3
configs are the portable variants used for released phase-3 cross-topology
evaluation.

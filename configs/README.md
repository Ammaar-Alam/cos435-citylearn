# config layout

These files are the shared config surface for the implemented benchmark paths.

- `env/` holds benchmark and path-level config
- `train/` holds method configs for RBC, PPO, and SAC
- `eval/` holds evaluation output settings
- `splits/` holds split names and intent, not downloaded data

The files here are live inputs to the repo runners:

- `train/rbc/rbc_builtin.yaml` drives the built-in RBC path
- `train/sac/*.yaml` drives the centralized and shared SAC runners
- `train/sac/*_smoke.yaml` are intentionally tiny configs for test coverage only
- `eval/default.yaml` is the normal full-eval path
- `eval/sac_smoke.yaml` disables simulation export and caps evaluation steps for fast SAC smoke runs

PPO remains a placeholder config contract. SAC configs are real and are launchable either through `scripts/train/run_sac.py`, the Make targets, or the dashboard runner registry for the exposed variants.

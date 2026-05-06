# cluster sweep scripts (Neuronic)

72-cell hyperparameter-first sweep for the final project: 3 algos x 8 lrs x 3 seeds.
Trains shared-parameter PPO, shared-parameter SAC, and shared-parameter TD3 on `public_dev`,
then evaluates each checkpoint on the released `phase_2_online_eval_{1,2,3}` and
`phase_3_{1,2,3}` splits.

## one-time setup

From the Neuronic login node:

```bash
ssh neuronic
bash <(curl -fsSL https://raw.githubusercontent.com/Ammaar-Alam/cos435-citylearn/main/scripts/cluster/setup_neuronic.sh)
# or, if you already cloned:
cd /u/$USER/cos435-citylearn
bash scripts/cluster/setup_neuronic.sh
```

`setup_neuronic.sh` clones the repo (if needed), installs the venv via
`uv`, pulls the CityLearn 2023 `public_dev`, released phase-2, and phase-3
datasets, validates the env schema, and provisions `results/sweep/` before any
direct `sbatch` call so SLURM's fallback `--output`/`--error` paths resolve on
a fresh checkout.

Override branches with `BRANCH=<branch-name> bash scripts/cluster/setup_neuronic.sh`.

Override the install root with `ROOT_DIR=/some/other/path bash scripts/cluster/setup_neuronic.sh`.
The shell-level pieces honor `${ROOT_DIR:-/u/${USER}/cos435-citylearn}` for
the checkout and `${SWEEP_ROOT}` for sweep artifacts.

`submit_sweep.sh` writes durable artifacts to
`/n/fs/pvl-lidar/cache/$USER/citylearn/sweeps/$SWEEP_ID` by default and
overrides SLURM stdout/stderr paths to land under that same sweep root. Direct
`sbatch scripts/cluster/sweep.slurm` still works, but then outputs fall back to
`$ROOT_DIR/results/sweep`.

Note: we use `/u/$USER/` (NFS-shared home) instead of `/scratch/` because
`/scratch` is node-local on Neuronic, so compute nodes can't see files
staged on the login node.

## launch the sweep

```bash
cd /u/$USER/cos435-citylearn
bash scripts/cluster/submit_sweep.sh
```

The wrapper is the canonical entry point: it creates the pvl-lidar sweep root,
exports `ROOT_DIR`/`SWEEP_ROOT`, and forwards to `sbatch scripts/cluster/sweep.slurm`.
Set `SWEEP_ID=lr-screen-r1` to choose a stable output directory. You can pass
extra flags through, eg. `JOB=rerun_evals ALGO=sac bash scripts/cluster/submit_sweep.sh`.

Calling `sbatch scripts/cluster/sweep.slurm` directly still works once
`results/sweep/` exists.

The array dispatches **72 tasks** (3 algos x 8 lrs x 3 seeds), capped at 72
concurrent tasks. Each task requests 4 CPUs and 12 GB RAM (no GPU; this
workload is mostly CPU-bound).
Array id decomposes as:

- `algo_idx = id / 24` -> 0=ppo, 1=sac, 2=td3
- `lr_idx   = (id%24)/3` -> `1e-5,3e-5,1e-4,2e-4,3e-4,5e-4,1e-3,3e-3`
- `seed_idx = id % 3` -> seeds `0,1,2`

Each cell trains on `public_dev`, saves a checkpoint, then evaluates released
phase-2 and phase-3 splits. Per-cell outputs land in
`$SWEEP_ROOT/<algo>_lr<lr>_seed<n>/{train,eval_phase_*}.json`; checkpoints and
metrics land under `$SWEEP_ROOT/{runs,metrics,manifests}`.

`results/sweep/` and pvl-lidar sweep roots are untracked raw artifacts. Commit
only normalized summary rows under `submission/results/`.

The full cap is 72 x 4 = 288 active CPUs.

## monitor

```bash
squeue -u $USER
tail -f /n/fs/pvl-lidar/cache/$USER/citylearn/sweeps/<sweep-id>/logs/slurm-<jobid>_<taskid>.out
```

## aggregate

When the array completes:

```bash
python scripts/cluster/aggregate_sweep.py \
  --sweep-root "$SWEEP_ROOT" \
  --out "$SWEEP_ROOT/summary.csv"
cat "$SWEEP_ROOT/summary.csv"
```

This fails loudly if any expected cell or split is missing. To force
a partial aggregation pass `--allow-missing`.

The CSV has one row per (algo, lr, seed, split) with the
`average_score` from that cell. Group by `(algo, lr, split)` for
mean +/- std across seeds. Pick the best `lr` per algo on `public_dev`,
then report the released phase-2 and phase-3 numbers of that configuration in
the final writeup.

## artifact boundary

The cluster scripts are the canonical Princeton/Neuronic path for the current
PPO/SAC/TD3 shared-parameter sweep. They do not cover the historical PPO phase-2
backfill path; see `scripts/analysis/README.md` for that specialized exporter.

Do not commit:

- `results/sweep/`
- Slurm stdout/stderr
- checkpoint directories
- downloaded dataset bundles
- Google Drive staging directories

After aggregation, copy only final CSV rows needed by the report into
`submission/results/`.

## targeted final sweep

After the broad learning-rate screen, run the smaller algorithm-specific matrix
with:

```bash
cd /u/$USER/cos435-citylearn
JOB=final_sweep SWEEP_ID=citylearn-final-hp-20260506-r1 bash scripts/cluster/submit_sweep.sh
```

This dispatches 81 cells:

- SAC: `lr x reward_scaling x seed`
- TD3: `lr x exploration_noise x seed`
- PPO: `lr x ent_coef x seed`

The default Slurm array cap is 75 cells x 4 CPUs = 300 CPUs.

Aggregate it with:

```bash
python scripts/cluster/aggregate_final_sweep.py \
  --sweep-root "$SWEEP_ROOT" \
  --out "$SWEEP_ROOT/summary.csv"
```

## MAPPO prototype smoke

Before adding MAPPO to any broad matrix, run one CPU-only smoke job:

```bash
cd /u/$USER/cos435-citylearn
JOB=mappo_smoke SWEEP_ID=mappo-prototype-YYYYMMDD-r1 bash scripts/cluster/submit_sweep.sh
```

This uses `configs/train/mappo/mappo_shared_ctde_smoke.yaml` and writes the
checkpoint, metrics, and manifest under the selected `/n/fs/pvl-lidar` sweep
root. It requests 4 CPUs and 12 GB RAM, matching the existing CPU-bound sweep
cell shape.

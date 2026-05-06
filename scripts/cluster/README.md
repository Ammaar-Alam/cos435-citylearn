# cluster sweep scripts (Neuronic)

60-cell robust sweep for the final project: 3 algos x 2 lrs x 10 seeds.
Trains shared-parameter PPO, shared-parameter SAC, and shared-parameter TD3 on `public_dev`,
then cross-topology evaluates each checkpoint on `phase_3_{1,2,3}`.

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
`uv`, pulls the CityLearn 2023 phase_2_local_evaluation (public_dev
training) and phase_3_{1,2,3} (cross-topology eval) datasets, validates
the env schema, and, importantly, provisions `results/sweep/` before
any `sbatch` call so SLURM's `--output`/`--error` paths resolve on a
fresh checkout.

Override branches with `BRANCH=<branch-name> bash scripts/cluster/setup_neuronic.sh`.

Override the install root with `ROOT_DIR=/some/other/path bash scripts/cluster/setup_neuronic.sh`. The
shell-level pieces (`setup_neuronic.sh`, `submit_sweep.sh`, the body of
`sweep.slurm` / `rerun_evals.slurm`, `run_cell.sh`, `rerun_eval_cell.sh`)
honor `${ROOT_DIR:-/u/${USER}/cos435-citylearn}`, so the checkout and
the per-cell JSON outputs track the override.

**One caveat:** `#SBATCH --output=` / `--error=` are parsed by SLURM
*before* the job body runs, and SLURM does not expand shell variables
there; only its own format codes (`%u`, `%A`, `%a`, `%j`). The two
`.slurm` drivers therefore hardcode
`/u/%u/cos435-citylearn/results/sweep/slurm-%A_%a.out` for stdout/stderr
even when `$ROOT_DIR` points elsewhere. If you're running from a
non-default root, either create a symlink (`ln -s $ROOT_DIR/results/sweep
/u/$USER/cos435-citylearn/results/sweep`) or edit the `#SBATCH`
directives to hardcode the new path.

Note: we use `/u/$USER/` (NFS-shared home) instead of `/scratch/` because
`/scratch` is node-local on Neuronic, so compute nodes can't see files
staged on the login node.

## launch the sweep

```bash
cd /u/$USER/cos435-citylearn
bash scripts/cluster/submit_sweep.sh
```

The wrapper is the canonical entry point: it runs `mkdir -p
"$ROOT_DIR/results/sweep"` (redundant with `setup_neuronic.sh` but safe
on re-submits) and forwards to `sbatch scripts/cluster/sweep.slurm`. You
can pass extra flags through, eg. `bash scripts/cluster/submit_sweep.sh
--export=ALGO=sac,ALL JOB=rerun_evals`.

Calling `sbatch scripts/cluster/sweep.slurm` directly still works once
`results/sweep/` exists.

The array dispatches **60 tasks** (3 algos x 2 lrs x 10 seeds), each
requesting 4 CPUs and 8 GB RAM (no GPU; this workload is CPU-bound).
Array id decomposes as:

- `algo_idx = id / 20` -> 0=ppo, 1=sac, 2=td3
- `lr_idx   = (id%20)/10` -> 0=1e-4, 1=3e-4
- `seed     = id % 10` -> 0..9

Each cell trains on `public_dev` (~50 min), saves a checkpoint, then
cross-topology evals on `phase_3_{1,2,3}`. Per-cell outputs land in
`results/sweep/<algo>_lr<lr>_seed<n>/{train,eval_phase_3_*}.json`.

`results/sweep/` is gitignored; the shared source of truth for sweep
outputs is the `COS 435/erik/...` Google Drive folder. Mirror each run's
JSONs and `summary.csv` there after aggregation. Commit only normalized
summary rows under `submission/results/`.

Total wall time should be ~50 min if enough CPUs are free
(60 x 4 = 240 CPUs; the cluster usually has >1000 idle).

## monitor

```bash
squeue -u $USER
tail -f results/sweep/slurm-<jobid>_<taskid>.out
```

## aggregate

When the array completes:

```bash
python scripts/cluster/aggregate_sweep.py
cat results/sweep/summary.csv
```

This fails loudly if any expected cell or split is missing. To force
a partial aggregation pass `--allow-missing`.

The CSV has one row per (algo, lr, seed, split) with the
`average_score` from that cell. Group by `(algo, lr, split)` for
mean +/- std across seeds. Pick the best `lr` per algo on `public_dev`,
then report the `phase_3_*` numbers of that configuration in the
final writeup.

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

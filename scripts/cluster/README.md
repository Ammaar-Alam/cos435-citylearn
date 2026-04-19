# cluster sweep scripts (Neuronic)

40-cell robust sweep for the final project: 2 algos × 2 lrs × 10 seeds.
Trains shared-parameter PPO and shared-parameter SAC on `public_dev`,
then cross-topology evaluates each checkpoint on `phase_3_{1,2,3}`.

## one-time setup

From the Neuronic login node:

```bash
ssh neuronic
git clone https://github.com/Ammaar-Alam/cos435-citylearn.git /u/$USER/cos435-citylearn
cd /u/$USER/cos435-citylearn
bash scripts/cluster/setup_neuronic.sh
```

By default this checks out `feat/ppo-baseline`. Override the branch
after merge with `BRANCH=main bash scripts/cluster/setup_neuronic.sh`.

The script installs the venv via `uv` (which downloads Python 3.10
automatically), pulls the CityLearn 2023 dataset, and validates the
env schema.

Note: we use `/u/$USER/` (NFS-shared home) instead of `/scratch/` because
`/scratch` is node-local on Neuronic — compute nodes can't see files
staged on the login node.

## launch the sweep

```bash
cd /u/$USER/cos435-citylearn
sbatch scripts/cluster/sweep.slurm
```

The array dispatches **40 tasks** (2 algos × 2 lrs × 10 seeds), each
requesting 4 CPUs and 8 GB RAM (no GPU — this workload is CPU-bound).
Array id decomposes as:

- `algo_idx = id / 20`   → 0=ppo, 1=sac
- `lr_idx   = (id%20)/10` → 0=1e-4, 1=3e-4
- `seed     = id % 10`    → 0..9

Each cell trains on `public_dev` (~50 min), saves a checkpoint, then
cross-topology evals on `phase_3_{1,2,3}`. Per-cell outputs land in
`results/sweep/<algo>_lr<lr>_seed<n>/{train,eval_phase_3_*}.json`.

Total wall time should be ~50 min if enough CPUs are free
(40 × 4 = 160 CPUs; the cluster usually has >1000 idle).

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

The CSV has one row per (algo, lr, seed, split) with the
`average_score` from that cell. Group by `(algo, lr, split)` for
mean±std across seeds. Pick the best `lr` per algo on `public_dev`,
then report the `phase_3_*` numbers of that configuration in the
final writeup.

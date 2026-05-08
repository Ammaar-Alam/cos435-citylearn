#!/usr/bin/env bash
#SBATCH --job-name=cos435-sweep
#SBATCH --array=0-71%72
#SBATCH --partition=all
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/slurm-%A_%a.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/slurm-%A_%a.out

# 72-cell sweep: 3 shared algos x 8 lrs x 3 seeds.
# array id decomposes as:
#   algo_idx = id / 24         (0 = ppo, 1 = sac, 2 = td3)
#   lr_idx   = (id % 24) / 3
#   seed_idx = id % 3          (0, 1, 2)
#
# submit: sbatch scripts/cluster/sweep.slurm
# monitor: squeue -u $USER; tail -f results/sweep/slurm-<jobid>_<taskid>.out
# aggregate after: python scripts/cluster/aggregate_sweep.py

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/sweep}"
mkdir -p "$SWEEP_ROOT"

ALGOS=(ppo sac td3)
LRS=(1e-5 3e-5 1e-4 2e-4 3e-4 5e-4 1e-3 3e-3)
SEEDS=(0 1 2)

TASK_ID=$SLURM_ARRAY_TASK_ID
ALGO_IDX=$(( TASK_ID / 24 ))
LR_IDX=$(( (TASK_ID % 24) / 3 ))
SEED_IDX=$(( TASK_ID % 3 ))

ALGO="${ALGOS[$ALGO_IDX]}"
LR="${LRS[$LR_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "task $TASK_ID: algo=$ALGO lr=$LR seed=$SEED sweep_root=$SWEEP_ROOT on $(hostname)"
bash "$ROOT_DIR/scripts/cluster/run_cell.sh" "$ALGO" "$LR" "$SEED"

#!/usr/bin/env bash
#SBATCH --job-name=cos435-sweep
#SBATCH --array=0-39
#SBATCH --partition=all
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/slurm-%A_%a.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/slurm-%A_%a.out

# 40-cell sweep: 2 algos x 2 lrs x 10 seeds.
# array id decomposes as:
#   algo_idx = id / 20         (0 = ppo, 1 = sac)
#   lr_idx   = (id % 20) / 10  (0 = 1e-4, 1 = 3e-4)
#   seed     = id % 10
#
# submit: sbatch scripts/cluster/sweep.slurm
# monitor: squeue -u $USER; tail -f results/sweep/slurm-<jobid>_<taskid>.out
# aggregate after: python scripts/cluster/aggregate_sweep.py

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
mkdir -p "$ROOT_DIR/results/sweep"

ALGOS=(ppo sac)
LRS=(1e-4 3e-4)

TASK_ID=$SLURM_ARRAY_TASK_ID
ALGO_IDX=$(( TASK_ID / 20 ))
LR_IDX=$(( (TASK_ID % 20) / 10 ))
SEED=$(( TASK_ID % 10 ))

ALGO="${ALGOS[$ALGO_IDX]}"
LR="${LRS[$LR_IDX]}"

echo "task $TASK_ID: algo=$ALGO lr=$LR seed=$SEED on $(hostname)"
bash "$ROOT_DIR/scripts/cluster/run_cell.sh" "$ALGO" "$LR" "$SEED"

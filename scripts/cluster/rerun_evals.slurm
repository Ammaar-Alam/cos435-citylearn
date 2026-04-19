#!/usr/bin/env bash
#SBATCH --job-name=cos435-rerun
#SBATCH --array=0-19
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/rerun-%A_%a.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/rerun-%A_%a.out

# re-runs phase_3_{1,2,3} evals for the 20 ppo cells whose first
# eval attempt failed because the datasets hadn't been downloaded yet.
# array decomposition:
#   lr_idx = TASK_ID / 10   (0=1e-4, 1=3e-4)
#   seed   = TASK_ID % 10

set -euo pipefail

ROOT_DIR="/u/${USER}/cos435-citylearn"

LRS=(1e-4 3e-4)
TASK_ID=$SLURM_ARRAY_TASK_ID
LR_IDX=$(( TASK_ID / 10 ))
SEED=$(( TASK_ID % 10 ))
LR="${LRS[$LR_IDX]}"

echo "rerun task $TASK_ID: algo=ppo lr=$LR seed=$SEED on $(hostname)"
bash "$ROOT_DIR/scripts/cluster/rerun_eval_cell.sh" ppo "$LR" "$SEED"

#!/usr/bin/env bash
#SBATCH --job-name=cos435-rerun
#SBATCH --array=0-23%24
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/rerun-%A_%a.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/rerun-%A_%a.out

# re-runs released phase-2 and phase-3 evals for the 24 cells of a single algo whose
# first eval attempt failed (e.g. because datasets weren't downloaded yet).
# default algo is ppo; override with: sbatch --export=ALGO=sac,ALL rerun_evals.slurm
# array decomposition:
#   lr_idx   = TASK_ID / 3
#   seed_idx = TASK_ID % 3

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/sweep}"
mkdir -p "$SWEEP_ROOT"
ALGO="${ALGO:-ppo}"

LRS=(1e-5 3e-5 1e-4 2e-4 3e-4 5e-4 1e-3 3e-3)
SEEDS=(0 1 2)
TASK_ID=$SLURM_ARRAY_TASK_ID
LR_IDX=$(( TASK_ID / 3 ))
SEED_IDX=$(( TASK_ID % 3 ))
LR="${LRS[$LR_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "rerun task $TASK_ID: algo=$ALGO lr=$LR seed=$SEED sweep_root=$SWEEP_ROOT on $(hostname)"
bash "$ROOT_DIR/scripts/cluster/rerun_eval_cell.sh" "$ALGO" "$LR" "$SEED"

#!/usr/bin/env bash
#SBATCH --job-name=cos435-residual-sac
#SBATCH --array=0-26%27
#SBATCH --partition=all
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/residual-sac-%A_%a.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/residual-sac-%A_%a.out

# 27-cell targeted sweep:
#   residual SAC: 3 lrs x 3 residual_scaling values x 3 seeds

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/residual_sac_sweep}"
mkdir -p "$SWEEP_ROOT"

SEEDS=(0 1 2)
LRS=(1e-4 3e-4 1e-3)
RESIDUAL_SCALINGS=(0.5 0.75 1.0)
RESIDUAL_EXPERT_POLICY="${RESIDUAL_EXPERT_POLICY:-adaptive_storage_v1}"

TASK_ID=$SLURM_ARRAY_TASK_ID
LR="${LRS[$(( TASK_ID / 9 ))]}"
RESIDUAL_SCALING="${RESIDUAL_SCALINGS[$(( (TASK_ID % 9) / 3 ))]}"
SEED="${SEEDS[$(( TASK_ID % 3 ))]}"

echo "task $TASK_ID: algo=sac_residual lr=$LR seed=$SEED residual_scaling=$RESIDUAL_SCALING expert_policy=$RESIDUAL_EXPERT_POLICY sweep_root=$SWEEP_ROOT on $(hostname)"
bash "$ROOT_DIR/scripts/cluster/run_final_cell.sh" \
  sac_residual \
  "$LR" \
  "$SEED" \
  residual_scaling \
  "$RESIDUAL_SCALING"

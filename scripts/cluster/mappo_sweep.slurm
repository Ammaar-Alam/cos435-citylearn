#!/usr/bin/env bash
#SBATCH --job-name=cos435-mappo
#SBATCH --array=0-17%9
#SBATCH --partition=all
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/mappo-%A_%a.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/mappo-%A_%a.out

# 18-cell MAPPO CTDE sweep:
#   3 learning rates x 2 entropy coefficients x 3 seeds.
# Kept separate from final_sweep.slurm so the existing PPO/SAC/TD3 array can
# finish and aggregate under its original 81-cell contract.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/mappo_sweep}"
mkdir -p "$SWEEP_ROOT"

LRS=(1e-4 3e-4 1e-3)
ENT_COEFS=(0.0 0.01)
SEEDS=(0 1 2)

TASK_ID=$SLURM_ARRAY_TASK_ID
LR="${LRS[$(( TASK_ID / 6 ))]}"
ENT_COEF="${ENT_COEFS[$(( (TASK_ID % 6) / 3 ))]}"
SEED="${SEEDS[$(( TASK_ID % 3 ))]}"

echo "task $TASK_ID: algo=mappo lr=$LR seed=$SEED ent_coef=$ENT_COEF sweep_root=$SWEEP_ROOT on $(hostname)"
bash "$ROOT_DIR/scripts/cluster/run_final_cell.sh" "mappo" "$LR" "$SEED" "ent_coef" "$ENT_COEF"

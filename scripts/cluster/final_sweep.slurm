#!/usr/bin/env bash
#SBATCH --job-name=cos435-final
#SBATCH --array=0-80%75
#SBATCH --partition=all
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/final-%A_%a.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/final-%A_%a.out

# 81-cell final targeted sweep, capped at 75 concurrent tasks / 300 CPUs:
#   SAC: 4 lrs x 3 reward_scaling values x 3 seeds = 36
#   TD3: 3 lrs x 3 exploration_noise values x 3 seeds = 27
#   PPO: 3 lrs x 2 ent_coef values x 3 seeds = 18

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/final_sweep}"
mkdir -p "$SWEEP_ROOT"

SEEDS=(0 1 2)
TASK_ID=$SLURM_ARRAY_TASK_ID

if (( TASK_ID < 36 )); then
  ALG="sac"
  LRS=(1e-4 2e-4 5e-4 1e-3)
  HPS=(2.5 5.0 10.0)
  LOCAL_ID=$TASK_ID
  HP_NAME="reward_scaling"
  LR="${LRS[$(( LOCAL_ID / 9 ))]}"
  HP_VALUE="${HPS[$(( (LOCAL_ID % 9) / 3 ))]}"
  SEED="${SEEDS[$(( LOCAL_ID % 3 ))]}"
elif (( TASK_ID < 63 )); then
  ALG="td3"
  LRS=(3e-4 5e-4 1e-3)
  HPS=(0.05 0.1 0.2)
  LOCAL_ID=$(( TASK_ID - 36 ))
  HP_NAME="exploration_noise"
  LR="${LRS[$(( LOCAL_ID / 9 ))]}"
  HP_VALUE="${HPS[$(( (LOCAL_ID % 9) / 3 ))]}"
  SEED="${SEEDS[$(( LOCAL_ID % 3 ))]}"
else
  ALG="ppo"
  LRS=(1e-4 1e-3 3e-3)
  HPS=(0.0 0.01)
  LOCAL_ID=$(( TASK_ID - 63 ))
  HP_NAME="ent_coef"
  LR="${LRS[$(( LOCAL_ID / 6 ))]}"
  HP_VALUE="${HPS[$(( (LOCAL_ID % 6) / 3 ))]}"
  SEED="${SEEDS[$(( LOCAL_ID % 3 ))]}"
fi

echo "task $TASK_ID: algo=$ALG lr=$LR seed=$SEED $HP_NAME=$HP_VALUE sweep_root=$SWEEP_ROOT on $(hostname)"
bash "$ROOT_DIR/scripts/cluster/run_final_cell.sh" "$ALG" "$LR" "$SEED" "$HP_NAME" "$HP_VALUE"

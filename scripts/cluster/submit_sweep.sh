#!/usr/bin/env bash
# canonical entry point for submitting the cross-algo sweep on neuronic.
# ensures the SLURM stdout/stderr sink exists before handing off to sbatch
# (SLURM opens --output/--error paths before the job body can mkdir them).
#
# usage:
#   bash scripts/cluster/submit_sweep.sh                  # submit sweep.slurm
#   JOB=rerun_evals bash scripts/cluster/submit_sweep.sh  # submit a different driver
#   SWEEP_ID=lr-screen-r1 \
#     bash scripts/cluster/submit_sweep.sh                # choose a stable output root
#
# extra flags after the script name are forwarded to sbatch, eg:
#   bash scripts/cluster/submit_sweep.sh --export=ALGO=sac,ALL

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
JOB="${JOB:-sweep}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d-%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-/n/fs/pvl-lidar/cache/${USER}/citylearn/sweeps/${SWEEP_ID}}"
LOG_ROOT="$SWEEP_ROOT/logs"

mkdir -p "$LOG_ROOT"
export ROOT_DIR SWEEP_ROOT

echo "submitting $JOB with SWEEP_ROOT=$SWEEP_ROOT"
exec sbatch \
  --output="$LOG_ROOT/slurm-%A_%a.out" \
  --error="$LOG_ROOT/slurm-%A_%a.out" \
  "$@" \
  "$ROOT_DIR/scripts/cluster/${JOB}.slurm"

#!/usr/bin/env bash
# canonical entry point for submitting the cross-algo sweep on neuronic.
# ensures the SLURM stdout/stderr sink exists before handing off to sbatch
# (SLURM opens --output/--error paths before the job body can mkdir them).
#
# usage:
#   bash scripts/cluster/submit_sweep.sh                  # submit sweep.slurm
#   JOB=rerun_evals bash scripts/cluster/submit_sweep.sh  # submit a different driver
#   ROOT_DIR=/scratch/$USER/cos435-citylearn \
#     bash scripts/cluster/submit_sweep.sh                # override install root
#
# extra flags after the script name are forwarded to sbatch, eg:
#   bash scripts/cluster/submit_sweep.sh --export=ALGO=sac,ALL

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
JOB="${JOB:-sweep}"

mkdir -p "$ROOT_DIR/results/sweep"

exec sbatch "$@" "$ROOT_DIR/scripts/cluster/${JOB}.slurm"

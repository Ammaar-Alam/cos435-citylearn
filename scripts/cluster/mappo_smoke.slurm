#!/usr/bin/env bash
#SBATCH --job-name=cos435-mappo-smoke
#SBATCH --partition=all
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/u/%u/cos435-citylearn/results/sweep/mappo-smoke-%j.out
#SBATCH --error=/u/%u/cos435-citylearn/results/sweep/mappo-smoke-%j.out

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-/n/fs/pvl-lidar/cache/${USER}/citylearn/sweeps/mappo-prototype-$(date +%Y%m%d-%H%M%S)}"
RUNS_ROOT="$SWEEP_ROOT/runs"
METRICS_ROOT="$SWEEP_ROOT/metrics"
MANIFESTS_ROOT="$SWEEP_ROOT/manifests"
UI_EXPORTS_ROOT="$SWEEP_ROOT/ui_exports"
mkdir -p "$RUNS_ROOT" "$METRICS_ROOT" "$MANIFESTS_ROOT" "$UI_EXPORTS_ROOT"

cd "$ROOT_DIR"
source .venv/bin/activate

export COS435_REQUIRE_DATA=1
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "MAPPO smoke on $(hostname)"
echo "ROOT_DIR=$ROOT_DIR"
echo "SWEEP_ROOT=$SWEEP_ROOT"

python scripts/train/run_mappo.py \
  --config configs/train/mappo/mappo_shared_ctde_smoke.yaml \
  --eval-config configs/eval/mappo_smoke.yaml \
  --output-root "$RUNS_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --manifests-root "$MANIFESTS_ROOT" \
  --ui-exports-root "$UI_EXPORTS_ROOT" \
  --artifacts-root "$SWEEP_ROOT" \
  | tee "$SWEEP_ROOT/train.json"

echo "MAPPO smoke done -> $SWEEP_ROOT"

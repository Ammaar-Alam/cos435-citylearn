#!/usr/bin/env bash
# run one sweep cell: train {algo} on public_dev with {lr, seed}, then
# eval the resulting checkpoint on released phase-2 and phase-3 splits. writes a per-cell
# summary JSON so the aggregation step can just glob the outputs.
#
# invoked by sweep.slurm with positional args:
#   $1 = algo (ppo|sac|td3)
#   $2 = lr (float, eg 3e-4)
#   $3 = seed (int)

set -euo pipefail

ALGO="${1:?usage: run_cell.sh <algo> <lr> <seed>}"
LR="${2:?usage: run_cell.sh <algo> <lr> <seed>}"
SEED="${3:?usage: run_cell.sh <algo> <lr> <seed>}"

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/sweep}"
cd "$ROOT_DIR"
source .venv/bin/activate

case "$ALGO" in
  ppo) TRAIN_SCRIPT="scripts/train/run_ppo.py"; CONFIG="configs/train/ppo/ppo_shared_dtde_reward_v2.yaml" ;;
  sac) TRAIN_SCRIPT="scripts/train/run_sac.py"; CONFIG="configs/train/sac/sac_shared_dtde_reward_v2.yaml" ;;
  td3) TRAIN_SCRIPT="scripts/train/run_td3.py"; CONFIG="configs/train/td3/td3_shared_dtde_reward_v2.yaml" ;;
  *)   echo "unknown algo: $ALGO" >&2; exit 2 ;;
esac

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/official_released.yaml}"
EVAL_SPLITS="${EVAL_SPLITS:-phase_2_online_eval_1 phase_2_online_eval_2 phase_2_online_eval_3 phase_3_1 phase_3_2 phase_3_3}"
read -r -a SPLITS <<< "$EVAL_SPLITS"

# cell id embeds lr and seed so results/sweep/<cell>/ is unique across the 72-cell matrix
CELL_ID="${ALGO}_lr${LR}_seed${SEED}"
SUMMARY_DIR="$SWEEP_ROOT/${CELL_ID}"
RUNS_ROOT="$SWEEP_ROOT/runs"
METRICS_ROOT="$SWEEP_ROOT/metrics"
MANIFESTS_ROOT="$SWEEP_ROOT/manifests"
UI_EXPORTS_ROOT="$SWEEP_ROOT/ui_exports"
mkdir -p "$SUMMARY_DIR" "$RUNS_ROOT" "$METRICS_ROOT" "$MANIFESTS_ROOT" "$UI_EXPORTS_ROOT"

export COS435_REQUIRE_DATA=1
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

echo "=== train $ALGO lr=$LR seed=$SEED on public_dev ==="
python "$TRAIN_SCRIPT" \
  --config "$CONFIG" \
  --eval-config "$EVAL_CONFIG" \
  --seed "$SEED" \
  --lr "$LR" \
  --output-root "$RUNS_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --manifests-root "$MANIFESTS_ROOT" \
  --ui-exports-root "$UI_EXPORTS_ROOT" \
  --artifacts-root "$SWEEP_ROOT" \
  | tee "$SUMMARY_DIR/train.json"

RUN_ID="$(python -c "import json,sys; print(json.load(open('$SUMMARY_DIR/train.json'))['run_id'])")"
echo "trained run_id=$RUN_ID"

for SPLIT in "${SPLITS[@]}"; do
  echo "=== cross-topology eval: $ALGO lr=$LR seed=$SEED on $SPLIT ==="
  python "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --eval-config "$EVAL_CONFIG" \
    --artifact-id "$RUN_ID" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --lr "$LR" \
    --output-root "$RUNS_ROOT" \
    --metrics-root "$METRICS_ROOT" \
    --manifests-root "$MANIFESTS_ROOT" \
    --ui-exports-root "$UI_EXPORTS_ROOT" \
    --artifacts-root "$SWEEP_ROOT" \
    | tee "$SUMMARY_DIR/eval_${SPLIT}.json"
done

echo "cell done: $CELL_ID -> $SUMMARY_DIR"

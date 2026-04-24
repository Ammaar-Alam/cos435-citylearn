#!/usr/bin/env bash
# run one sweep cell: train {algo} on public_dev with {lr, seed}, then
# eval the resulting checkpoint on phase_3_{1,2,3}. writes a per-cell
# summary JSON so the aggregation step can just glob the outputs.
#
# invoked by sweep.slurm with positional args:
#   $1 = algo (ppo|sac)
#   $2 = lr (float, eg 3e-4)
#   $3 = seed (int)

set -euo pipefail

ALGO="${1:?usage: run_cell.sh <algo> <lr> <seed>}"
LR="${2:?usage: run_cell.sh <algo> <lr> <seed>}"
SEED="${3:?usage: run_cell.sh <algo> <lr> <seed>}"

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
cd "$ROOT_DIR"
source .venv/bin/activate

case "$ALGO" in
  ppo) TRAIN_SCRIPT="scripts/train/run_ppo.py"; CONFIG="configs/train/ppo/ppo_shared_dtde_reward_v2.yaml" ;;
  sac) TRAIN_SCRIPT="scripts/train/run_sac.py"; CONFIG="configs/train/sac/sac_shared_dtde_reward_v2.yaml" ;;
  *)   echo "unknown algo: $ALGO" >&2; exit 2 ;;
esac

EVAL_CONFIG="configs/eval/default.yaml"
# cell id embeds lr and seed so results/sweep/<cell>/ is unique across the 40-cell matrix
CELL_ID="${ALGO}_lr${LR}_seed${SEED}"
SUMMARY_DIR="$ROOT_DIR/results/sweep/${CELL_ID}"
mkdir -p "$SUMMARY_DIR"

export COS435_REQUIRE_DATA=1
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

echo "=== train $ALGO lr=$LR seed=$SEED on public_dev ==="
python "$TRAIN_SCRIPT" \
  --config "$CONFIG" \
  --eval-config "$EVAL_CONFIG" \
  --seed "$SEED" \
  --lr "$LR" \
  | tee "$SUMMARY_DIR/train.json"

RUN_ID="$(python -c "import json,sys; print(json.load(open('$SUMMARY_DIR/train.json'))['run_id'])")"
echo "trained run_id=$RUN_ID"

for SPLIT in phase_3_1 phase_3_2 phase_3_3; do
  echo "=== cross-topology eval: $ALGO lr=$LR seed=$SEED on $SPLIT ==="
  python "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --eval-config "$EVAL_CONFIG" \
    --artifact-id "$RUN_ID" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --lr "$LR" \
    | tee "$SUMMARY_DIR/eval_${SPLIT}.json"
done

echo "cell done: $CELL_ID -> $SUMMARY_DIR"

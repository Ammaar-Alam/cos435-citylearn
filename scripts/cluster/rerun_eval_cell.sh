#!/usr/bin/env bash
# rerun cross-topology evals for a cell whose train.json already exists.
# uses the saved checkpoint (run_id embedded in train.json).
#
# args: $1=algo, $2=lr, $3=seed

set -euo pipefail

ALGO="${1:?usage: rerun_eval_cell.sh <algo> <lr> <seed>}"
LR="${2:?usage: rerun_eval_cell.sh <algo> <lr> <seed>}"
SEED="${3:?usage: rerun_eval_cell.sh <algo> <lr> <seed>}"

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
cd "$ROOT_DIR"
source .venv/bin/activate

case "$ALGO" in
  ppo) TRAIN_SCRIPT="scripts/train/run_ppo.py"; CONFIG="configs/train/ppo/ppo_shared_dtde_reward_v2.yaml" ;;
  sac) TRAIN_SCRIPT="scripts/train/run_sac.py"; CONFIG="configs/train/sac/sac_shared_dtde_reward_v2.yaml" ;;
  *)   echo "unknown algo: $ALGO" >&2; exit 2 ;;
esac

EVAL_CONFIG="configs/eval/default.yaml"
CELL_ID="${ALGO}_lr${LR}_seed${SEED}"
SUMMARY_DIR="$ROOT_DIR/results/sweep/${CELL_ID}"

if [[ ! -f "$SUMMARY_DIR/train.json" ]]; then
  echo "missing $SUMMARY_DIR/train.json — nothing to eval" >&2
  exit 2
fi

export COS435_REQUIRE_DATA=1
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

RUN_ID="$(python -c "import json; print(json.load(open('$SUMMARY_DIR/train.json'))['run_id'])")"
echo "rerun evals for $CELL_ID (run_id=$RUN_ID)"

for SPLIT in phase_3_1 phase_3_2 phase_3_3; do
  OUT="$SUMMARY_DIR/eval_${SPLIT}.json"
  if [[ -f "$OUT" ]] && python -c "import json; json.load(open('$OUT'))['average_score']" >/dev/null 2>&1; then
    echo "$CELL_ID/$SPLIT already complete; skipping"
    continue
  fi
  echo "=== eval $CELL_ID on $SPLIT ==="
  python "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --eval-config "$EVAL_CONFIG" \
    --artifact-id "$RUN_ID" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --lr "$LR" \
    | tee "$OUT"
done

echo "rerun done: $CELL_ID"

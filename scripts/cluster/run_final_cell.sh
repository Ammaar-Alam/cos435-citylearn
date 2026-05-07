#!/usr/bin/env bash
# Run one targeted final-sweep cell. The final matrix varies one algorithm-specific
# hyperparameter in addition to learning rate and seed.
#
# args:
#   $1 = algo (ppo|sac|sac_residual|td3)
#   $2 = lr
#   $3 = seed
#   $4 = hyperparameter name (ent_coef|reward_scaling|exploration_noise)
#   $5 = hyperparameter value

set -euo pipefail

ALGO="${1:?usage: run_final_cell.sh <algo> <lr> <seed> <hp_name> <hp_value>}"
LR="${2:?usage: run_final_cell.sh <algo> <lr> <seed> <hp_name> <hp_value>}"
SEED="${3:?usage: run_final_cell.sh <algo> <lr> <seed> <hp_name> <hp_value>}"
HP_NAME="${4:?usage: run_final_cell.sh <algo> <lr> <seed> <hp_name> <hp_value>}"
HP_VALUE="${5:?usage: run_final_cell.sh <algo> <lr> <seed> <hp_name> <hp_value>}"

ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/final_sweep}"
cd "$ROOT_DIR"
source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

EXPERT_POLICY_META=""

case "$ALGO:$HP_NAME" in
  ppo:ent_coef)
    TRAIN_SCRIPT="scripts/train/run_ppo.py"
    CONFIG="configs/train/ppo/ppo_shared_dtde_reward_v2.yaml"
    EXTRA_ARGS=(--ent-coef "$HP_VALUE")
    ;;
  sac:reward_scaling)
    TRAIN_SCRIPT="scripts/train/run_sac.py"
    CONFIG="configs/train/sac/sac_shared_dtde_reward_v2.yaml"
    EXTRA_ARGS=(--reward-scaling "$HP_VALUE")
    ;;
  sac_residual:residual_scaling)
    TRAIN_SCRIPT="scripts/train/run_sac.py"
    CONFIG="configs/train/sac/sac_shared_residual_adaptive_reward_v2.yaml"
    RESIDUAL_EXPERT_POLICY="${RESIDUAL_EXPERT_POLICY:-adaptive_storage_v1}"
    EXPERT_POLICY_META="$RESIDUAL_EXPERT_POLICY"
    EXTRA_ARGS=(--expert-policy "$RESIDUAL_EXPERT_POLICY" --residual-scaling "$HP_VALUE")
    ;;
  td3:exploration_noise)
    TRAIN_SCRIPT="scripts/train/run_td3.py"
    CONFIG="configs/train/td3/td3_shared_dtde_reward_v2.yaml"
    EXTRA_ARGS=(--exploration-noise "$HP_VALUE")
    ;;
  *)
    echo "unsupported final-sweep cell: algo=$ALGO hp=$HP_NAME" >&2
    exit 2
    ;;
esac

label_value() {
  printf '%s' "$1" | sed -e 's/-/m/g' -e 's/\./p/g'
}

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/official_released.yaml}"
EVAL_SPLITS="${EVAL_SPLITS:-phase_2_online_eval_1 phase_2_online_eval_2 phase_2_online_eval_3 phase_3_1 phase_3_2 phase_3_3}"
read -r -a SPLITS <<< "$EVAL_SPLITS"

HP_LABEL="$(label_value "$HP_VALUE")"
CELL_ID="${ALGO}_lr${LR}_hp-${HP_NAME}_val-${HP_LABEL}_seed${SEED}"
SUMMARY_DIR="$SWEEP_ROOT/${CELL_ID}"
RUNS_ROOT="$SWEEP_ROOT/runs"
METRICS_ROOT="$SWEEP_ROOT/metrics"
MANIFESTS_ROOT="$SWEEP_ROOT/manifests"
UI_EXPORTS_ROOT="$SWEEP_ROOT/ui_exports"
mkdir -p "$SUMMARY_DIR" "$RUNS_ROOT" "$METRICS_ROOT" "$MANIFESTS_ROOT" "$UI_EXPORTS_ROOT"

META_EXTRA=""
if [ -n "$EXPERT_POLICY_META" ]; then
  META_EXTRA=",\"expert_policy\":\"$EXPERT_POLICY_META\""
fi

cat > "$SUMMARY_DIR/meta.json" <<EOF
{"cell_id":"$CELL_ID","algo":"$ALGO","lr":"$LR","seed":$SEED,"hyperparameter":"$HP_NAME","hyperparameter_value":"$HP_VALUE"$META_EXTRA}
EOF

export COS435_REQUIRE_DATA=1
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

echo "=== train $CELL_ID on public_dev ==="
python "$TRAIN_SCRIPT" \
  --config "$CONFIG" \
  --eval-config "$EVAL_CONFIG" \
  --seed "$SEED" \
  --lr "$LR" \
  "${EXTRA_ARGS[@]}" \
  --output-root "$RUNS_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --manifests-root "$MANIFESTS_ROOT" \
  --ui-exports-root "$UI_EXPORTS_ROOT" \
  --artifacts-root "$SWEEP_ROOT" \
  | tee "$SUMMARY_DIR/train.json"

RUN_ID="$(python -c "import json; print(json.load(open('$SUMMARY_DIR/train.json'))['run_id'])")"
echo "trained run_id=$RUN_ID"

for SPLIT in "${SPLITS[@]}"; do
  echo "=== eval $CELL_ID on $SPLIT ==="
  python "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --eval-config "$EVAL_CONFIG" \
    --artifact-id "$RUN_ID" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --lr "$LR" \
    "${EXTRA_ARGS[@]}" \
    --output-root "$RUNS_ROOT" \
    --metrics-root "$METRICS_ROOT" \
    --manifests-root "$MANIFESTS_ROOT" \
    --ui-exports-root "$UI_EXPORTS_ROOT" \
    --artifacts-root "$SWEEP_ROOT" \
    | tee "$SUMMARY_DIR/eval_${SPLIT}.json"
done

echo "cell done: $CELL_ID -> $SUMMARY_DIR"

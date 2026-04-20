#!/usr/bin/env bash
# one-time bootstrap on neuronic: clone repo, install venv, pull dataset
# run from the login node: bash scripts/cluster/setup_neuronic.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Ammaar-Alam/cos435-citylearn.git}"
BRANCH="${BRANCH:-main}"
ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "cloning $REPO_URL -> $ROOT_DIR"
  git clone "$REPO_URL" "$ROOT_DIR"
fi

# SLURM opens --output/--error paths before the job body runs, so the sink
# must exist before any sbatch call (including on a fresh checkout).
mkdir -p "$ROOT_DIR/results/sweep"

cd "$ROOT_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# uv venv --python 3.10 will download python if the system doesn't have it
bash scripts/setup/install_env.sh requirements/benchmark.txt

source .venv/bin/activate
# training uses phase_2_local_evaluation (default); sweep cross-topology evals
# need the three phase_3 datasets, so fetch them explicitly here
python scripts/setup/download_citylearn_2023.py \
  --dataset citylearn_challenge_2023_phase_2_local_evaluation \
  --dataset citylearn_challenge_2023_phase_3_1 \
  --dataset citylearn_challenge_2023_phase_3_2 \
  --dataset citylearn_challenge_2023_phase_3_3
make env-schema

echo
echo "setup complete. next: bash scripts/cluster/submit_sweep.sh"

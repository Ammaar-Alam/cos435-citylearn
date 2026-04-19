#!/usr/bin/env bash
# one-time bootstrap on neuronic: clone repo, install venv, pull dataset
# run from the login node: bash scripts/cluster/setup_neuronic.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Ammaar-Alam/cos435-citylearn.git}"
BRANCH="${BRANCH:-feat/ppo-baseline}"
ROOT_DIR="${ROOT_DIR:-/u/${USER}/cos435-citylearn}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "cloning $REPO_URL -> $ROOT_DIR"
  git clone "$REPO_URL" "$ROOT_DIR"
fi

cd "$ROOT_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# uv venv --python 3.10 will download python if the system doesn't have it
bash scripts/setup/install_env.sh requirements/benchmark.txt

source .venv/bin/activate
make download-citylearn
make env-schema

echo
echo "setup complete. next: sbatch scripts/cluster/sweep.slurm"

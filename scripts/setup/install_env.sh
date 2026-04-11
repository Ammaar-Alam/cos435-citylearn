#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="${1:-requirements/dev.txt}"

cd "$ROOT_DIR"

if command -v uv >/dev/null 2>&1; then
  uv venv --seed --python 3.10 .venv
elif command -v python3.10 >/dev/null 2>&1; then
  python3.10 -m venv .venv
else
  echo "python 3.10 is required for this repo" >&2
  echo "install uv or python3.10, then rerun this script" >&2
  exit 1
fi

source .venv/bin/activate
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade pip
python -m pip install -r "$REQ_FILE"
python -m pip install -e .

echo
echo "env is ready"
echo "source .venv/bin/activate"
echo "make env-info"
echo "make check"

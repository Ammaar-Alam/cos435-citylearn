#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "benchmark environment is not ready" >&2
  echo "run `make install-benchmark` first" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/setup/download_citylearn_2023.py "$@"

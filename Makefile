PYTHON := .venv/bin/python
MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: install install-benchmark test check env-info repo-tree download-citylearn env-schema smoke train-rbc

install:
	bash scripts/setup/install_env.sh

install-benchmark:
	bash scripts/setup/install_env.sh requirements/benchmark.txt

test:
	$(PYTHON) -m pytest -q --ignore=tests/smoke

check:
	$(PYTHON) scripts/check/check_configs.py
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q --ignore=tests/smoke

env-info:
	$(PYTHON) scripts/setup/env_info.py

download-citylearn:
	bash scripts/setup/download_citylearn_2023.sh

env-schema:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/inspect/write_env_schema.py

smoke:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) -m pytest -q -m smoke tests/smoke

train-rbc:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/train/run_rbc.py

repo-tree:
	find . -maxdepth 3 -type f | sort

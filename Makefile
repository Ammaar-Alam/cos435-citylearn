PYTHON := .venv/bin/python

.PHONY: install test check env-info repo-tree

install:
	bash scripts/setup/install_env.sh

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) scripts/check/check_configs.py
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q

env-info:
	$(PYTHON) scripts/setup/env_info.py

repo-tree:
	find . -maxdepth 3 -type f | sort

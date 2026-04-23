PYTHON := .venv/bin/python
MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
NPM_CACHE := $(CURDIR)/.npm-cache

.PHONY: install install-benchmark test check env-info repo-tree download-citylearn download-citylearn-all env-schema smoke train-rbc train-ppo train-sac train-sac-shared submission-results figures check-ui-exports dashboard-install dashboard-build dashboard-check dashboard-backend dashboard-frontend ui ui-open

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
	@if [ -d apps/dashboard/node_modules ]; then $(MAKE) dashboard-check; else echo "dashboard build skipped; run make dashboard-install first"; fi

env-info:
	$(PYTHON) scripts/setup/env_info.py

download-citylearn:
	bash scripts/setup/download_citylearn_2023.sh

download-citylearn-all:
	bash scripts/setup/download_citylearn_2023.sh --all

env-schema:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/inspect/write_env_schema.py

smoke:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) -m pytest -q -m smoke tests/smoke

train-rbc:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/train/run_rbc.py

train-sac:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/train/run_sac.py --config configs/train/sac/sac_central_baseline.yaml --eval-config configs/eval/default.yaml

train-ppo:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/train/run_ppo.py --config configs/train/ppo/ppo_central_baseline.yaml --eval-config configs/eval/default.yaml

train-sac-shared:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/train/run_sac.py --config configs/train/sac/sac_shared_dtde_reward_v2.yaml --eval-config configs/eval/default.yaml

submission-results:
	$(PYTHON) scripts/analysis/export_submission_results.py

figures:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/analysis/make_figures.py

check-ui-exports:
	$(PYTHON) scripts/check/validate_official_ui_exports.py

dashboard-install:
	cd apps/dashboard && npm_config_cache="$(NPM_CACHE)" npm ci

dashboard-build:
	cd apps/dashboard && npm run build

dashboard-check:
	cd apps/dashboard && npm run build

dashboard-backend:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/dashboard/run_backend.py

dashboard-frontend:
	cd apps/dashboard && npm run dev

ui:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/dashboard/run_ui.py

ui-open:
	COS435_REQUIRE_DATA=1 MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/dashboard/run_ui.py --open-browser

repo-tree:
	find . -maxdepth 3 -type f | sort

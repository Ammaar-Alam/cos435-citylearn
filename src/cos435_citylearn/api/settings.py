from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from cos435_citylearn.paths import CONFIGS_DIR, REPO_ROOT, RESULTS_DIR


@dataclass(frozen=True)
class ApiSettings:
    repo_root: Path = REPO_ROOT
    config_root: Path = CONFIGS_DIR
    results_root: Path = RESULTS_DIR
    run_root: Path = RESULTS_DIR / "runs"
    manifests_root: Path = RESULTS_DIR / "manifests"
    ui_exports_root: Path = RESULTS_DIR / "ui_exports"
    jobs_root: Path = RESULTS_DIR / "dashboard" / "jobs"
    imported_artifacts_root: Path = RESULTS_DIR / "dashboard" / "artifacts"
    artifacts_root: Path = RESULTS_DIR
    frontend_root: Path = REPO_ROOT / "apps" / "dashboard"
    frontend_dist: Path = REPO_ROOT / "apps" / "dashboard" / "dist"
    python_executable: Path = Path(sys.executable)
    mpl_config_dir: Path = REPO_ROOT / ".cache" / "matplotlib"
    max_concurrent_jobs: int = 1


SETTINGS = ApiSettings()

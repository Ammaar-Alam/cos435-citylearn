import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from cos435_citylearn.paths import REPO_ROOT

DEFAULT_PACKAGES = [
    "CityLearn",
    "stable-baselines3",
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "PyYAML",
    "pytest",
    "ruff",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def maybe_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def package_versions(names: list[str] | None = None) -> dict[str, str]:
    selected = DEFAULT_PACKAGES if names is None else names
    versions = {}

    for name in selected:
        value = maybe_version(name)
        if value is not None:
            versions[name] = value

    return versions


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            cwd=REPO_ROOT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def build_environment_lock(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
    }

    if extra:
        payload.update(extra)

    return payload

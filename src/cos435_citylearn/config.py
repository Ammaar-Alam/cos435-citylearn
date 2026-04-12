from pathlib import Path
from typing import Any

import yaml

from cos435_citylearn.paths import REPO_ROOT


def resolve_path(path_like: str | Path, base: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    anchor = REPO_ROOT if base is None else Path(base)
    return (anchor / path).resolve()


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    data = yaml.safe_load(path.read_text())
    return {} if data is None else data

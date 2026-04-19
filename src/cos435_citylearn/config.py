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


def assert_training_allowed_on_split(split_config: dict[str, Any], *, artifact_id: str | None) -> None:
    """Reject training runs on held-out / tuning-disabled splits.

    Reads the `split:` block of a split YAML. If `held_out` is true (or
    `tuning_allowed` is false) and no `artifact_id` is set, raise. An
    `artifact_id` means the runner is in eval-only mode (no training),
    which is legitimate on held-out splits.
    """
    split_block = split_config.get("split", {}) if split_config else {}
    held_out = bool(split_block.get("held_out", False))
    tuning_allowed = bool(split_block.get("tuning_allowed", True))

    if artifact_id is None and (held_out or not tuning_allowed):
        name = split_block.get("name", "<unknown>")
        raise ValueError(
            f"refusing to train on split '{name}' because it is marked held_out/tuning_allowed=false. "
            "provide an artifact_id to run in eval-only mode, or switch to a non-held-out split."
        )

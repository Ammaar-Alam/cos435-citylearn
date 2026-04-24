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


def assert_training_allowed_on_split(
    split_config: dict[str, Any],
    *,
    artifact_id: str | None,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Reject training runs on held-out / tuning-disabled splits.

    Reads the `split:` block of a split YAML. If `held_out` is true (or
    `tuning_allowed` is false) and the runner is in training mode
    (no artifact_id, no checkpoint_path), raise. Either an ``artifact_id``
    (dashboard-imported or locally-trained run) or a ``checkpoint_path``
    (direct on-disk checkpoint, e.g. ``scripts/eval/run_sac_checkpoint.py``)
    signals eval-only mode (no training), which is legitimate on held-out
    splits like the phase-3 post-competition datasets.
    """
    split_block = split_config.get("split", {}) if split_config else {}
    held_out = bool(split_block.get("held_out", False))
    tuning_allowed = bool(split_block.get("tuning_allowed", True))

    eval_mode = artifact_id is not None or checkpoint_path is not None
    if not eval_mode and (held_out or not tuning_allowed):
        name = split_block.get("name", "<unknown>")
        raise ValueError(
            f"refusing to train on split '{name}' because it is marked "
            "held_out/tuning_allowed=false. provide an artifact_id or "
            "checkpoint_path to run in eval-only mode, or switch to a "
            "non-held-out split."
        )

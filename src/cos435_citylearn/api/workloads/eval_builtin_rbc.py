from __future__ import annotations

from pathlib import Path
from typing import Any

from cos435_citylearn.baselines.rbc import run_rbc


def run(request: dict[str, Any], context) -> dict[str, Any]:
    return run_rbc(
        config_path=request["config_path"],
        eval_config_path=request["eval_config_path"],
        artifact_id=request.get("artifact_id"),
        job_id=request["job_id"],
        job_dir=Path(request["job_dir"]),
        progress_context=context,
    )

from __future__ import annotations

from pathlib import Path
from typing import Any

from cos435_citylearn.baselines.ppo import run_ppo


def run(request: dict[str, Any], context) -> dict[str, Any]:
    return run_ppo(
        config_path=request["config_path"],
        eval_config_path=request["eval_config_path"],
        output_root=request.get("output_root"),
        metrics_root=request.get("metrics_root"),
        manifests_root=request.get("manifests_root"),
        artifacts_root=request.get("artifacts_root"),
        imported_artifacts_root=request.get("imported_artifacts_root"),
        artifact_id=request.get("artifact_id"),
        job_id=request["job_id"],
        job_dir=Path(request["job_dir"]),
        progress_context=context,
    )

from __future__ import annotations

from pathlib import Path
from typing import Any

from cos435_citylearn.baselines.sac import run_sac


def run(request: dict[str, Any], context) -> dict[str, Any]:
    return run_sac(
        config_path=request["config_path"],
        eval_config_path=request["eval_config_path"],
        output_root=request.get("output_root"),
        metrics_root=request.get("metrics_root"),
        manifests_root=request.get("manifests_root"),
        ui_exports_root=request.get("ui_exports_root"),
        artifacts_root=request.get("artifacts_root"),
        imported_artifacts_root=request.get("imported_artifacts_root"),
        artifact_id=request.get("artifact_id"),
        job_id=request["job_id"],
        job_dir=Path(request["job_dir"]),
        progress_context=context,
        split_override=request.get("split_override"),
        seed_override=request.get("seed_override"),
        lr_override=request.get("lr_override"),
        allow_cross_reward_eval=bool(request.get("allow_cross_reward_eval", False)),
    )

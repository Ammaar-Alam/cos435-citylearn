import os
from datetime import datetime, timezone
from uuid import uuid4


def _format_lr(lr: float) -> str:
    # keep path-safe (no dots), readable for humans scanning dir listings
    return f"{lr:.6g}".replace(".", "p").replace("+", "").replace("-", "m")


def _resolve_job_id() -> str:
    # SLURM --requeue keeps the same JOB_ID/TASK_ID across attempts, so without
    # SLURM_RESTART_COUNT a requeued task would overwrite the failed attempt's
    # run_id. Append it when present and non-zero.
    restart = os.environ.get("SLURM_RESTART_COUNT")
    restart_suffix = f".r{restart}" if restart and restart != "0" else ""
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"{array_job}.{array_task}{restart_suffix}"
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        return f"{job}{restart_suffix}"
    return uuid4().hex[:8]


def build_run_id(
    algo: str,
    variant: str,
    split: str,
    seed: int,
    now: datetime | None = None,
    lr: float | None = None,
    *,
    job_id: str | None = None,
) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    lr_part = f"__lr{_format_lr(lr)}" if lr is not None else ""
    if job_id is None:
        job_id = _resolve_job_id()
    return f"{algo}__{variant}__{split}__seed{seed}{lr_part}__{stamp}__{job_id}"

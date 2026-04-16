from __future__ import annotations

from cos435_citylearn.api.workloads.eval_builtin_rbc import run as run_eval_builtin_rbc
from cos435_citylearn.api.workloads.eval_ppo_checkpoint import run as run_eval_ppo_checkpoint
from cos435_citylearn.api.workloads.eval_sac_checkpoint import run as run_eval_sac_checkpoint

WORKLOADS = {
    "eval_builtin_rbc": run_eval_builtin_rbc,
    "eval_ppo_checkpoint": run_eval_ppo_checkpoint,
    "eval_sac_checkpoint": run_eval_sac_checkpoint,
}


def get_workload(workload_id: str):
    if workload_id not in WORKLOADS:
        raise KeyError(f"unknown workload: {workload_id}")
    return WORKLOADS[workload_id]

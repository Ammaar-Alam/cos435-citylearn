from __future__ import annotations

from cos435_citylearn.api.workloads.eval_builtin_rbc import run as run_eval_builtin_rbc

WORKLOADS = {
    "eval_builtin_rbc": run_eval_builtin_rbc,
}


def get_workload(workload_id: str):
    if workload_id not in WORKLOADS:
        raise KeyError(f"unknown workload: {workload_id}")
    return WORKLOADS[workload_id]

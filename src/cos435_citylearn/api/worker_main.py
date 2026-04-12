from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from cos435_citylearn.api.services.job_event_store import JobEventStore
from cos435_citylearn.api.services.job_state_store import JobStateStore
from cos435_citylearn.api.workloads import WorkloadContext, get_workload
from cos435_citylearn.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.job_file).read_text())
    result_path = Path(request["result_path"])
    error_path = Path(request["error_path"])
    job_dir = Path(args.job_file).parent
    jobs_root = job_dir.parent
    context = WorkloadContext(
        job_id=request["job_id"],
        job_dir=job_dir,
        state_store=JobStateStore(jobs_root),
        event_store=JobEventStore(jobs_root),
        job_kind=request.get("job_kind", "evaluation"),
    )

    try:
        workload = get_workload(request["workload_id"])
        payload = workload(request, context)
        write_json(result_path, payload)
        context.finish(result=payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(error_path, error_payload)
        context.fail(error_message=str(exc))
        print(error_payload["traceback"], flush=True)
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from cos435_citylearn.api.schemas import JobSummary, LaunchJobRequest

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(request: Request) -> list[JobSummary]:
    return request.app.state.job_manager.list_jobs()


@router.post("")
def create_job(request: Request, payload: LaunchJobRequest) -> JobSummary:
    try:
        return request.app.state.job_manager.submit(payload)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{job_id}")
def get_job(request: Request, job_id: str) -> JobSummary:
    try:
        return request.app.state.job_manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> JobSummary:
    try:
        return request.app.state.job_manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{job_id}/logs")
def get_logs(
    request: Request,
    job_id: str,
    tail: int = Query(default=200, ge=20, le=1000),
) -> dict[str, str]:
    try:
        logs = request.app.state.job_manager.tail_logs(job_id, tail=tail)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job_id": job_id, "logs": logs}

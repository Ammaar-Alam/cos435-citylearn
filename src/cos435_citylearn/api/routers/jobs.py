from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from cos435_citylearn.api.schemas import (
    JobEvent,
    JobState,
    JobSummary,
    LaunchJobRequest,
    PlaybackResponse,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(request: Request) -> list[JobSummary]:
    return request.app.state.job_manager.list_jobs()


@router.post("")
def create_job(request: Request, payload: LaunchJobRequest) -> JobSummary:
    try:
        return request.app.state.job_manager.submit(payload)
    except (FileNotFoundError, KeyError, ValueError) as exc:
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


@router.get("/{job_id}/state")
def get_state(request: Request, job_id: str) -> JobState:
    try:
        return JobState(**request.app.state.job_manager.get_state(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{job_id}/events")
def get_events(
    request: Request,
    job_id: str,
    after_seq: int = Query(default=0, ge=0),
) -> list[JobEvent]:
    try:
        events = request.app.state.job_manager.get_events(job_id, after_seq=after_seq)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [JobEvent(**event) for event in events]


@router.get("/{job_id}/preview")
def get_preview(request: Request, job_id: str) -> PlaybackResponse:
    try:
        return request.app.state.playback_store.get_job_preview(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{job_id}/artifacts")
def get_artifacts(request: Request, job_id: str) -> list[dict[str, str]]:
    try:
        return request.app.state.job_manager.list_artifacts(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

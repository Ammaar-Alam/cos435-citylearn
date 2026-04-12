from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from cos435_citylearn.api.schemas import PlaybackResponse, RunDetail, RunSummary

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("")
def list_runs(request: Request) -> list[RunSummary]:
    return request.app.state.run_store.list_runs()


@router.get("/{run_id}")
def get_run(request: Request, run_id: str) -> RunDetail:
    try:
        return request.app.state.run_store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/playback")
def get_playback(
    request: Request,
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=256, ge=32, le=1024),
) -> PlaybackResponse:
    try:
        return request.app.state.playback_store.get_playback(run_id, offset=offset, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

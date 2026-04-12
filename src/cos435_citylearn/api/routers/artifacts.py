from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from cos435_citylearn.api.schemas import (
    ArtifactDetail,
    ArtifactSummary,
    EvaluateArtifactRequest,
    JobSummary,
    LaunchJobRequest,
    PlaybackResponse,
)

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("")
def list_artifacts(request: Request) -> list[ArtifactSummary]:
    return request.app.state.artifact_store.list_artifacts()


@router.get("/{artifact_id}")
def get_artifact(request: Request, artifact_id: str) -> ArtifactDetail:
    try:
        return request.app.state.artifact_store.get_artifact(artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{artifact_id}/playback")
def get_artifact_playback(request: Request, artifact_id: str) -> PlaybackResponse:
    try:
        artifact = request.app.state.artifact_store.get_artifact(artifact_id)
        if not artifact.playback_path:
            raise ValueError("artifact does not include a playback payload")
        return request.app.state.playback_store.get_artifact_playback(artifact.playback_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/import")
async def import_artifact(
    request: Request,
    file: UploadFile = File(...),
    artifact_kind: str = Form(...),
    label: str = Form(""),
    notes: str | None = Form(default=None),
    runner_id: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
) -> ArtifactDetail:
    try:
        return await request.app.state.artifact_store.import_upload(
            file=file,
            artifact_kind=artifact_kind,
            label=label,
            notes=notes,
            runner_id=runner_id,
            algorithm=algorithm,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{artifact_id}/evaluate")
def evaluate_artifact(
    request: Request,
    artifact_id: str,
    payload: EvaluateArtifactRequest,
) -> JobSummary:
    try:
        launch_payload = request.app.state.artifact_store.build_evaluation_request(
            artifact_id, payload
        )
        launch_request = LaunchJobRequest(**launch_payload)
        return request.app.state.job_manager.submit(launch_request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

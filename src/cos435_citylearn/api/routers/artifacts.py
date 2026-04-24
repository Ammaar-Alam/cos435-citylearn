from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from cos435_citylearn.api.schemas import (
    ArtifactDetail,
    ArtifactKind,
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
    # ``extra_files`` lets a single POST ship companion files into the same
    # artifact directory. This is the only way to import a centralized PPO
    # checkpoint -- it needs ``ppo_model.zip`` (primary) + ``vec_normalize.pkl``
    # (companion) co-located so the loader can pair them by ``artifact_id``.
    # Optional; single-file uploads (SAC .pt, shared-PPO .pt, playback JSON)
    # continue to work by simply omitting this field.
    extra_files: list[UploadFile] = File(default_factory=list),
    artifact_kind: ArtifactKind = Form(...),
    label: str = Form(""),
    notes: str | None = Form(default=None),
    runner_id: str | None = Form(default=None),
    algorithm: str | None = Form(default=None),
) -> ArtifactDetail:
    try:
        return await request.app.state.artifact_store.import_upload(
            file=file,
            extra_files=extra_files,
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
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

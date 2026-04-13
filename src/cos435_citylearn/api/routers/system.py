from __future__ import annotations

import json

from fastapi import APIRouter, Request

from cos435_citylearn.api.services.runner_registry import list_runners
from cos435_citylearn.runtime import build_environment_lock

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/runners")
def runners() -> list[dict[str, object]]:
    return list_runners()


@router.get("/schema")
def schema(request: Request) -> dict[str, object]:
    schema_path = request.app.state.settings.manifests_root / "observation_action_schema.json"
    if not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text())


@router.get("/env")
def environment(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    dataset_manifest = settings.repo_root / "data" / "manifests" / "citylearn_2023_manifest.json"
    default_schema = (
        settings.repo_root
        / "data"
        / "external"
        / "citylearn_2023"
        / "citylearn_challenge_2023_phase_2_local_evaluation"
        / "schema.json"
    )
    return build_environment_lock(
        {
            "repo_root": str(settings.repo_root),
            "default_schema_present": default_schema.exists(),
            "dataset_manifest_present": dataset_manifest.exists(),
        }
    )

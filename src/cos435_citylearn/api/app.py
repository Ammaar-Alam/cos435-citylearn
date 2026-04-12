from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cos435_citylearn.api.routers.artifacts import router as artifacts_router
from cos435_citylearn.api.routers.jobs import router as jobs_router
from cos435_citylearn.api.routers.runs import router as runs_router
from cos435_citylearn.api.routers.system import router as system_router
from cos435_citylearn.api.services.artifact_store import ArtifactStore
from cos435_citylearn.api.services.job_manager import JobManager
from cos435_citylearn.api.services.playback_store import PlaybackStore
from cos435_citylearn.api.services.run_store import RunStore
from cos435_citylearn.api.settings import SETTINGS, ApiSettings


def _bind_services(app: FastAPI, settings: ApiSettings) -> None:
    app.state.settings = settings
    app.state.job_manager = JobManager(settings)
    app.state.run_store = RunStore(settings)
    app.state.playback_store = PlaybackStore(settings)
    app.state.artifact_store = ArtifactStore(settings)


def create_app(settings: ApiSettings = SETTINGS) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _bind_services(app, settings)
        yield

    app = FastAPI(title="COS435 CityLearn Dashboard", lifespan=lifespan)
    _bind_services(app, settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system_router)
    app.include_router(jobs_router)
    app.include_router(runs_router)
    app.include_router(artifacts_router)

    artifacts_root = settings.artifacts_root
    if artifacts_root.exists():
        app.mount("/artifacts", StaticFiles(directory=artifacts_root), name="artifacts")

    frontend_dist = settings.frontend_dist
    if frontend_dist.exists():
        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=assets_dir),
                name="assets",
            )

        @app.get("/dashboard/{path:path}", include_in_schema=False)
        def dashboard(path: str) -> FileResponse:
            candidate = frontend_dist / path
            if path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    else:

        @app.get("/dashboard", include_in_schema=False)
        def dashboard_unbuilt() -> JSONResponse:
            return JSONResponse(
                {
                    "status": "frontend-not-built",
                    "message": (
                        "run `make dashboard-install` and `make dashboard-build`, "
                        "or use `make ui` for local development"
                    ),
                }
            )

    @app.get("/", include_in_schema=False)
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "name": "COS435 CityLearn Dashboard API",
                "dashboard_path": "/dashboard",
                "api_root": "/api/system/health",
            }
        )

    return app

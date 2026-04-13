from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import cos435_citylearn.api.app as app_module
from cos435_citylearn.api.app import create_app
from cos435_citylearn.api.settings import ApiSettings
from cos435_citylearn.paths import CONFIGS_DIR, REPO_ROOT


def build_test_settings(tmp_path: Path) -> ApiSettings:
    frontend_root = tmp_path / "dashboard"
    frontend_dist = frontend_root / "dist"
    frontend_dist.mkdir(parents=True, exist_ok=True)
    (frontend_dist / "index.html").write_text("<html><body>dashboard</body></html>")
    (frontend_dist / "app.js").write_text("console.log('ok');")

    return ApiSettings(
        repo_root=tmp_path,
        config_root=CONFIGS_DIR,
        results_root=tmp_path / "results",
        run_root=tmp_path / "results" / "runs",
        manifests_root=tmp_path / "results" / "manifests",
        ui_exports_root=tmp_path / "results" / "ui_exports",
        jobs_root=tmp_path / "results" / "dashboard" / "jobs",
        imported_artifacts_root=tmp_path / "results" / "dashboard" / "artifacts",
        artifacts_root=tmp_path / "results",
        frontend_root=frontend_root,
        frontend_dist=frontend_dist,
        python_executable=REPO_ROOT / ".venv" / "bin" / "python",
        mpl_config_dir=tmp_path / ".cache" / "matplotlib",
        max_concurrent_jobs=1,
    )


def test_dashboard_route_serves_built_files_and_spa_fallback(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    asset_response = client.get("/dashboard/app.js")
    assert asset_response.status_code == 200
    assert "console.log('ok');" in asset_response.text

    route_response = client.get("/dashboard/runs/example")
    assert route_response.status_code == 200
    assert "dashboard" in route_response.text


def test_dashboard_route_rejects_path_traversal(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("do not leak")
    client = TestClient(create_app(settings))

    response = client.get("/dashboard/%2E%2E/%2E%2E/secret.txt")
    assert response.status_code == 404
    assert "do not leak" not in response.text


def test_dashboard_route_returns_404_for_missing_asset_like_paths(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.get("/dashboard/assets/missing-hash.js")

    assert response.status_code == 404


def test_create_app_binds_services_only_once(tmp_path: Path, monkeypatch) -> None:
    settings = build_test_settings(tmp_path)
    bind_count = 0
    original_bind = app_module._bind_services

    def counted_bind(app, bound_settings):
        nonlocal bind_count
        bind_count += 1
        return original_bind(app, bound_settings)

    monkeypatch.setattr(app_module, "_bind_services", counted_bind)

    app = create_app(settings)
    with TestClient(app):
        pass

    assert bind_count == 1

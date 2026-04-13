from fastapi.testclient import TestClient

from cos435_citylearn.api.app import create_app


def test_system_runners_endpoint_lists_current_catalog() -> None:
    client = TestClient(create_app())
    response = client.get("/api/system/runners")
    assert response.status_code == 200
    payload = response.json()
    assert any(item["runner_id"] == "rbc_builtin" for item in payload)
    assert any(item["runner_id"] == "sac_central_baseline" for item in payload)

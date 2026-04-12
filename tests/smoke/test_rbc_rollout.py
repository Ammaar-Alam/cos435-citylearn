import pytest

from cos435_citylearn.baselines import run_rbc
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


@pytest.mark.smoke
def test_rbc_rollout(tmp_path):
    require_benchmark_runtime()
    require_dataset()
    payload = run_rbc(
        output_root=tmp_path / "runs",
        metrics_root=tmp_path / "metrics",
        manifests_root=tmp_path / "manifests",
    )

    assert payload["average_score"] is not None
    assert (tmp_path / "manifests" / "environment_lock.json").exists()
    assert (tmp_path / "metrics" / f"{payload['run_id']}.csv").exists()

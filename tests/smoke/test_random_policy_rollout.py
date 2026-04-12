import pytest

from cos435_citylearn.smoke import run_random_rollout
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


@pytest.mark.smoke
def test_random_policy_rollout(tmp_path):
    require_benchmark_runtime()
    require_dataset()
    output_path = tmp_path / "random_rollout_trace.json"
    payload = run_random_rollout(max_steps=24, trace_output_path=output_path)

    assert payload["steps"] == 24
    assert payload["no_nan_observations"] is True
    assert payload["no_nan_rewards"] is True
    assert len(payload["trace"]) == 24
    assert output_path.exists()

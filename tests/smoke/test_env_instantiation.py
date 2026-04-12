import pytest

from cos435_citylearn.dataset import DEFAULT_DATASET_NAME
from cos435_citylearn.env import (
    CentralizedEnvAdapter,
    make_citylearn_env,
    write_env_schema_manifest,
)
from tests.smoke.helpers import require_benchmark_runtime, require_dataset


@pytest.mark.smoke
def test_env_instantiation_and_schema_export(tmp_path):
    require_benchmark_runtime()
    schema_path = require_dataset()
    bundle = make_citylearn_env()

    assert bundle.dataset_name == DEFAULT_DATASET_NAME
    assert bundle.schema_path == schema_path
    assert bundle.central_agent is True
    assert len(bundle.env.buildings) > 0
    assert len(bundle.env.action_space) == 1
    assert len(bundle.env.observation_space) == 1

    payload = write_env_schema_manifest(
        schema_output_path=tmp_path / "observation_action_schema.json",
        environment_lock_path=tmp_path / "environment_lock.json",
    )
    assert payload["dataset_name"] == DEFAULT_DATASET_NAME
    assert payload["action_dimensions"][0] == len(bundle.env.action_names[0])


@pytest.mark.smoke
def test_action_scaling_identity():
    require_benchmark_runtime()
    require_dataset()
    bundle = make_citylearn_env()
    adapter = CentralizedEnvAdapter(bundle.env)
    zero_action = [[0.0 for _ in bundle.env.action_names[0]]]

    assert adapter.clip_actions(zero_action) == zero_action

    for bound in adapter.action_bounds():
        assert all(value <= 1.0 for value in bound["high"])
        assert all(value >= -1.0 for value in bound["low"])

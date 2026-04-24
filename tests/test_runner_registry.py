from pathlib import Path

import yaml

from cos435_citylearn.api.schemas import LaunchJobRequest
from cos435_citylearn.api.services.runner_registry import RUNNERS, materialize_runner_files


def test_materialize_runner_files_applies_safe_overrides(tmp_path: Path) -> None:
    config_path, eval_config_path = materialize_runner_files(
        LaunchJobRequest(
            runner_id="rbc_builtin",
            seed=7,
            trace_limit=48,
            capture_render_frames=False,
            render_frame_width=720,
        ),
        job_dir=tmp_path,
    )
    config = yaml.safe_load(config_path.read_text())
    eval_config = yaml.safe_load(eval_config_path.read_text())

    assert config["training"]["seed"] == 7
    assert config["evaluation"]["trace_limit"] == 48
    assert eval_config["evaluation"]["capture_render_frames"] is False
    assert eval_config["evaluation"]["render_frame_width"] == 720


def test_sac_runners_are_launchable_and_checkpoint_evaluable() -> None:
    assert RUNNERS["sac_central_baseline"].launchable is True
    assert RUNNERS["sac_central_baseline"].supports_checkpoint_eval is True
    assert RUNNERS["sac_shared_dtde_reward_v2"].launchable is True
    assert RUNNERS["sac_shared_dtde_reward_v2"].supports_checkpoint_eval is True


def test_ppo_runners_are_launchable_and_checkpoint_evaluable() -> None:
    assert RUNNERS["ppo_central_baseline"].launchable is True
    assert RUNNERS["ppo_central_baseline"].supports_checkpoint_eval is True
    assert RUNNERS["ppo_shared_dtde_reward_v2"].launchable is True
    assert RUNNERS["ppo_shared_dtde_reward_v2"].supports_checkpoint_eval is True

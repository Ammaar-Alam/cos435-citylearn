from __future__ import annotations

from pathlib import Path

import yaml

from cos435_citylearn.baselines import run_rbc


class FakeProgressContext:
    def __init__(self) -> None:
        self.started = []
        self.updates = []
        self.artifacts = []

    def start(self, **payload) -> None:
        self.started.append(payload)

    def update(self, **payload) -> None:
        self.updates.append(payload)

    def artifact(self, **payload) -> None:
        self.artifacts.append(payload)


def test_run_rbc_emits_live_preview_updates(tmp_path: Path) -> None:
    eval_config = yaml.safe_load(Path("configs/eval/default.yaml").read_text())
    eval_config["evaluation"]["capture_render_frames"] = False
    eval_config["evaluation"]["max_render_frames"] = 8
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))
    progress = FakeProgressContext()

    payload = run_rbc(
        eval_config_path=eval_path,
        progress_context=progress,
        job_id="job_test_preview",
        job_dir=tmp_path / "job_test_preview",
    )

    assert progress.started
    assert progress.updates
    assert progress.artifacts
    assert payload["run_id"].startswith("rbc__")

    last_update = progress.updates[-1]
    preview_payload = last_update["preview_payload"]
    assert preview_payload["run_id"] == payload["run_id"]
    assert preview_payload["trace"]
    assert len(preview_payload["trace"]) <= 96
    assert preview_payload["episode_total_steps"] >= preview_payload["decision_steps"]
    assert preview_payload["trace"][-1]["step"] == preview_payload["preview_step"]
    assert preview_payload["window_start_step"] == preview_payload["preview_step"] + 1 - preview_payload["decision_steps"]

    artifact_kinds = {artifact["kind"] for artifact in progress.artifacts}
    assert {"playback", "simulation_export"} <= artifact_kinds


def test_run_rbc_bounds_preview_trace_when_export_disabled(tmp_path: Path) -> None:
    eval_config = yaml.safe_load(Path("configs/eval/default.yaml").read_text())
    eval_config["evaluation"]["capture_render_frames"] = False
    eval_config["evaluation"]["max_render_frames"] = 8
    eval_config["evaluation"]["export_simulation_data"] = False
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))
    progress = FakeProgressContext()

    payload = run_rbc(
        eval_config_path=eval_path,
        progress_context=progress,
        job_id="job_test_preview_no_export",
        job_dir=tmp_path / "job_test_preview_no_export",
    )

    assert progress.updates
    last_update = progress.updates[-1]["preview_payload"]
    assert last_update["trace"]
    assert len(last_update["trace"]) <= 96
    assert last_update["trace"][-1]["step"] == last_update["preview_step"]
    assert last_update["window_start_step"] == last_update["preview_step"] + 1 - last_update["decision_steps"]
    assert "simulation_dir" not in payload
    assert not progress.artifacts

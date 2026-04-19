from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cos435_citylearn.api.schemas import LaunchJobRequest
from cos435_citylearn.api.settings import SETTINGS
from cos435_citylearn.config import load_yaml, resolve_path


@dataclass(frozen=True)
class RunnerSpec:
    runner_id: str
    label: str
    algorithm: str
    variant: str
    description: str
    callable_path: str | None
    workload_id: str
    config_path: str
    eval_config_path: str
    supports_checkpoint_eval: bool = False

    @property
    def launchable(self) -> bool:
        return self.callable_path is not None


RUNNERS: dict[str, RunnerSpec] = {
    "rbc_builtin": RunnerSpec(
        runner_id="rbc_builtin",
        label="Built-in RBC",
        algorithm="rbc",
        variant="basic_rbc",
        description="CityLearn built-in rule-based controller on the official local-eval split",
        callable_path="cos435_citylearn.baselines.rbc.run_rbc",
        workload_id="eval_builtin_rbc",
        config_path="configs/train/rbc/rbc_builtin.yaml",
        eval_config_path="configs/eval/default.yaml",
    ),
    "ppo_central_baseline": RunnerSpec(
        runner_id="ppo_central_baseline",
        label="Centralized PPO",
        algorithm="ppo",
        variant="central_baseline",
        description="Config contract only. The PPO runner has not been implemented yet.",
        callable_path=None,
        workload_id="eval_ppo_checkpoint",
        config_path="configs/train/ppo/ppo_central_baseline.yaml",
        eval_config_path="configs/eval/default.yaml",
    ),
    "sac_central_baseline": RunnerSpec(
        runner_id="sac_central_baseline",
        label="Centralized SAC",
        algorithm="sac",
        variant="central_baseline",
        description="Centralized native-SAC baseline on the official local-eval split",
        callable_path="cos435_citylearn.baselines.sac.run_sac",
        workload_id="eval_sac_checkpoint",
        config_path="configs/train/sac/sac_central_baseline.yaml",
        eval_config_path="configs/eval/default.yaml",
        supports_checkpoint_eval=True,
    ),
    "sac_shared_dtde_reward_v2": RunnerSpec(
        runner_id="sac_shared_dtde_reward_v2",
        label="Shared SAC Reward v2",
        algorithm="sac",
        variant="shared_dtde_reward_v2",
        description="Parameter-shared decentralized SAC with district-context features and reward_v2",
        callable_path="cos435_citylearn.baselines.sac.run_sac",
        workload_id="eval_sac_checkpoint",
        config_path="configs/train/sac/sac_shared_dtde_reward_v2.yaml",
        eval_config_path="configs/eval/default.yaml",
        supports_checkpoint_eval=True,
    ),
    "ppo_shared_dtde_reward_v2": RunnerSpec(
        runner_id="ppo_shared_dtde_reward_v2",
        label="Shared PPO Reward v2",
        algorithm="ppo",
        variant="ppo_shared_dtde_reward_v2",
        description="Parameter-shared decentralized PPO with count-invariant district context and reward_v2",
        callable_path="cos435_citylearn.baselines.ppo.run_ppo",
        workload_id="eval_ppo_checkpoint",
        config_path="configs/train/ppo/ppo_shared_dtde_reward_v2.yaml",
        eval_config_path="configs/eval/default.yaml",
        supports_checkpoint_eval=True,
    ),
}


def list_runners() -> list[dict[str, Any]]:
    payload = []

    for spec in RUNNERS.values():
        payload.append(
            {
                "runner_id": spec.runner_id,
                "label": spec.label,
                "algorithm": spec.algorithm,
                "variant": spec.variant,
                "description": spec.description,
                "config_path": spec.config_path,
                "eval_config_path": spec.eval_config_path,
                "launchable": spec.launchable,
                "supports_checkpoint_eval": spec.supports_checkpoint_eval,
            }
        )

    return payload


def get_runner(runner_id: str) -> RunnerSpec:
    if runner_id not in RUNNERS:
        raise KeyError(f"unknown runner: {runner_id}")

    return RUNNERS[runner_id]


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def materialize_runner_files(
    request: LaunchJobRequest,
    *,
    job_dir: Path,
) -> tuple[Path, Path]:
    spec = get_runner(request.runner_id)
    config = load_yaml(spec.config_path)
    eval_config = load_yaml(spec.eval_config_path)

    if request.seed is not None:
        config["training"]["seed"] = int(request.seed)
    if request.split is not None:
        config["env"]["split"] = request.split
    if request.trace_limit is not None:
        config["evaluation"]["trace_limit"] = int(request.trace_limit)
    if request.capture_render_frames is not None:
        eval_config["evaluation"]["capture_render_frames"] = bool(request.capture_render_frames)
    if request.max_render_frames is not None:
        eval_config["evaluation"]["max_render_frames"] = int(request.max_render_frames)
    if request.render_frame_width is not None:
        eval_config["evaluation"]["render_frame_width"] = int(request.render_frame_width)

    config_path = _write_yaml(job_dir / "config.yaml", config)
    eval_config_path = _write_yaml(job_dir / "eval.yaml", eval_config)

    resolved_config_path = resolve_path(config_path)
    resolved_eval_path = resolve_path(eval_config_path)
    if (
        SETTINGS.config_root not in resolved_config_path.parents
        and job_dir not in resolved_config_path.parents
    ):
        raise ValueError("resolved config path escaped the allowed config roots")
    if (
        SETTINGS.config_root not in resolved_eval_path.parents
        and job_dir not in resolved_eval_path.parents
    ):
        raise ValueError("resolved eval config path escaped the allowed config roots")

    return config_path, eval_config_path

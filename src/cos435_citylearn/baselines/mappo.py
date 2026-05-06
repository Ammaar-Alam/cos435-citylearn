from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from cos435_citylearn.algorithms.mappo import (
    CentralizedMAPPOController,
    safe_load_mappo_checkpoint_payload,
    validate_mappo_checkpoint_env_compatibility,
    validate_mappo_checkpoint_runner_compatibility,
)
from cos435_citylearn.algorithms.mappo.features import CENTRALIZED_CRITIC_CONTEXT_VERSION
from cos435_citylearn.algorithms.ppo.shared_features import SHARED_CONTEXT_V2_DIMENSION
from cos435_citylearn.algorithms.sac.rewards import resolve_reward_function
from cos435_citylearn.baselines.ppo import (
    _resolved_artifact_path,
    _run_shared_ppo_evaluation_loop,
    _run_shared_ppo_training_loop,
    _write_shared_ppo_training_curve,
)
from cos435_citylearn.config import assert_training_allowed_on_split, load_yaml, resolve_path
from cos435_citylearn.env import make_citylearn_env
from cos435_citylearn.eval import flatten_metrics_row
from cos435_citylearn.io import ensure_parent, write_csv_row, write_json
from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.run_id import build_run_id
from cos435_citylearn.runtime import build_environment_lock, utc_now_iso


def _mappo_controller_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    features = config.get("features", {})
    return {
        "hidden_dimension": list(training.get("hidden_dimension", [64, 64])),
        "critic_hidden_dimension": list(
            training.get("critic_hidden_dimension", training.get("hidden_dimension", [64, 64]))
        ),
        "lr": float(training.get("learning_rate", 3e-4)),
        "clip_range": float(training.get("clip_range", 0.2)),
        "n_epochs": int(training.get("n_epochs", 10)),
        "minibatch_size": int(training.get("minibatch_size", 64)),
        "gamma": float(training.get("gamma", 0.99)),
        "gae_lambda": float(training.get("gae_lambda", 0.95)),
        "ent_coef": training.get("ent_coef", 0.01),
        "vf_coef": float(training.get("vf_coef", 0.5)),
        "max_grad_norm": float(training.get("max_grad_norm", 0.5)),
        "rollout_steps": int(training.get("rollout_steps", 2048)),
        "reward_scaling": float(training.get("reward_scaling", 1.0)),
        "shared_context_dimension": int(
            features.get("shared_context_dimension", SHARED_CONTEXT_V2_DIMENSION)
        ),
        "shared_context_version": str(features.get("shared_context_version", "v2")),
        "critic_context_version": str(
            features.get("critic_context_version", CENTRALIZED_CRITIC_CONTEXT_VERSION)
        ),
        "normalize_observations": bool(training.get("normalize_observations", True)),
        "normalize_critic_observations": bool(
            training.get("normalize_critic_observations", True)
        ),
        "normalize_rewards": bool(training.get("normalize_rewards", True)),
        "normalize_advantage": bool(training.get("normalize_advantage", True)),
        "target_kl": (None if training.get("target_kl") is None else float(training["target_kl"])),
    }


def _build_mappo_checkpoint_payload(
    *,
    run_id: str,
    config: dict[str, Any],
    env_bundle: Any,
    controller: CentralizedMAPPOController,
    training_step: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "algorithm": config["algorithm"]["name"],
        "variant": config["algorithm"]["variant"],
        "control_mode": config["algorithm"]["control_mode"],
        "reward_version": config["reward"]["version"],
        "features_version": config["features"].get("version"),
        "run_id": run_id,
        "training_step": int(training_step),
        "seed": int(config["training"]["seed"]),
        "dataset_name": env_bundle.dataset_name,
        "schema_path": str(env_bundle.schema_path),
        "generated_at": utc_now_iso(),
        "observation_names": env_bundle.env.observation_names,
        "action_names": env_bundle.env.action_names,
        "config": config,
        "controller_state": controller.checkpoint_state(),
    }


def _instantiate_mappo_from_checkpoint(
    env: Any,
    checkpoint_payload: dict[str, Any],
) -> CentralizedMAPPOController:
    controller_state = checkpoint_payload["controller_state"]
    controller = CentralizedMAPPOController(
        env,
        hidden_dimension=list(controller_state["hidden_dimension"]),
        critic_hidden_dimension=list(controller_state["critic_hidden_dimension"]),
        lr=float(controller_state["lr"]),
        clip_range=float(controller_state["clip_range"]),
        n_epochs=int(controller_state["n_epochs"]),
        minibatch_size=int(controller_state["minibatch_size"]),
        gamma=float(controller_state["gamma"]),
        gae_lambda=float(controller_state["gae_lambda"]),
        ent_coef=(
            controller_state["ent_coef_schedule"]
            if controller_state.get("ent_coef_schedule") is not None
            else float(controller_state["ent_coef"])
        ),
        vf_coef=float(controller_state["vf_coef"]),
        max_grad_norm=float(controller_state["max_grad_norm"]),
        rollout_steps=int(controller_state["rollout_steps"]),
        reward_scaling=float(controller_state["reward_scaling"]),
        shared_context_dimension=int(controller_state["shared_context_dimension"]),
        shared_context_version=str(controller_state["shared_context_version"]),
        critic_context_version=str(controller_state["critic_context_version"]),
        normalize_observations=bool(controller_state.get("normalize_observations", True)),
        normalize_critic_observations=bool(
            controller_state.get("normalize_critic_observations", True)
        ),
        normalize_rewards=bool(controller_state.get("normalize_rewards", True)),
        normalize_advantage=bool(controller_state.get("normalize_advantage", True)),
        target_kl=(
            None
            if controller_state.get("target_kl") is None
            else float(controller_state["target_kl"])
        ),
    )
    controller.load_checkpoint_state(controller_state)
    return controller


def _load_imported_mappo_checkpoint(
    *,
    artifact_id: str,
    imported_artifacts_root: str | Path | None,
    artifacts_root: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    if imported_artifacts_root is None:
        root = Path(artifacts_root) if artifacts_root is not None else RESULTS_DIR
        checkpoint_path = root / "runs" / artifact_id / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MAPPO checkpoint not found for artifact: {checkpoint_path}")
    else:
        if artifacts_root is None:
            raise ValueError("artifacts_root must be set when imported_artifacts_root is provided")
        from cos435_citylearn.algorithms.sac.checkpoints import resolve_imported_checkpoint_path

        checkpoint_path = resolve_imported_checkpoint_path(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
    checkpoint_payload = safe_load_mappo_checkpoint_payload(checkpoint_path)
    return checkpoint_path, checkpoint_payload


def run_mappo(
    config_path: str | Path = "configs/train/mappo/mappo_shared_ctde_reward_v2.yaml",
    eval_config_path: str | Path = "configs/eval/default.yaml",
    output_root: str | Path | None = None,
    metrics_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    ui_exports_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    imported_artifacts_root: str | Path | None = None,
    artifact_id: str | None = None,
    job_id: str | None = None,
    job_dir: str | Path | None = None,
    progress_context: Any | None = None,
    split_override: str | None = None,
    seed_override: int | None = None,
    lr_override: float | None = None,
    ent_coef_override: float | None = None,
    allow_cross_reward_eval: bool = False,
    **kwargs,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_config = load_yaml(eval_config_path)

    if split_override is not None:
        config["env"]["split"] = split_override
    if seed_override is not None:
        config["training"]["seed"] = int(seed_override)
    if lr_override is not None:
        config["training"]["learning_rate"] = float(lr_override)
    if ent_coef_override is not None:
        config["training"]["ent_coef"] = float(ent_coef_override)

    if config["algorithm"].get("name") != "mappo":
        raise ValueError("run_mappo requires algorithm.name=mappo")
    if config["algorithm"].get("control_mode") != "shared_ctde":
        raise ValueError("run_mappo requires algorithm.control_mode=shared_ctde")

    split_config_path = f"configs/splits/{config['env']['split']}.yaml"
    split_config = load_yaml(split_config_path)
    assert_training_allowed_on_split(split_config, artifact_id=artifact_id)

    reward_function = resolve_reward_function(config["reward"]["version"])
    env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        split_config_path,
        seed=config["training"]["seed"],
        central_agent=False,
        reward_function=reward_function,
    )
    variant = config["algorithm"]["variant"]
    run_id = build_run_id(
        algo=config["algorithm"]["name"],
        variant=variant,
        split=config["env"]["split"],
        seed=env_bundle.seed,
        lr=float(config["training"]["learning_rate"]),
    )
    run_root = RESULTS_DIR / "runs" if output_root is None else Path(output_root)
    metrics_dir = (
        resolve_path(eval_config["logging"]["metrics_root"])
        if metrics_root is None
        else Path(metrics_root)
    )
    manifests_dir = (
        resolve_path(eval_config["logging"]["manifests_root"])
        if manifests_root is None
        else Path(manifests_root)
    )
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "checkpoint.pt"
    training_curve_path = run_dir / "training_curve.csv"

    label_mismatches: dict[str, tuple[Any, Any]] = {}
    if artifact_id is None:
        training_controller = CentralizedMAPPOController(
            env_bundle.env,
            **_mappo_controller_kwargs(config),
        )
        curve_rows = _run_shared_ppo_training_loop(
            controller=training_controller,
            env_bundle=env_bundle,
            config=config,
            progress_context=progress_context,
        )
        checkpoint_payload = _build_mappo_checkpoint_payload(
            run_id=run_id,
            config=config,
            env_bundle=env_bundle,
            controller=training_controller,
            training_step=int(config["training"]["total_timesteps"]),
        )
        ensure_parent(checkpoint_path)
        torch.save(checkpoint_payload, checkpoint_path)
        _write_shared_ppo_training_curve(training_curve_path, curve_rows)
    else:
        checkpoint_path, checkpoint_payload = _load_imported_mappo_checkpoint(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
        label_mismatches = validate_mappo_checkpoint_runner_compatibility(
            checkpoint_payload,
            config,
            allow_cross_reward_eval=allow_cross_reward_eval,
        )
        _write_shared_ppo_training_curve(training_curve_path, [])

    if artifact_id is None:
        checkpoint_payload = safe_load_mappo_checkpoint_payload(checkpoint_path)

    eval_reward_function = resolve_reward_function(config["reward"]["version"])
    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=False,
        reward_function=eval_reward_function,
    )
    validate_mappo_checkpoint_env_compatibility(
        checkpoint_payload,
        observation_names=eval_env_bundle.env.observation_names,
        action_names=eval_env_bundle.env.action_names,
    )
    evaluation_controller = _instantiate_mappo_from_checkpoint(
        eval_env_bundle.env,
        checkpoint_payload,
    )
    metrics_payload, ui_export_payload, step_count, rollout_trace = _run_shared_ppo_evaluation_loop(
        controller=evaluation_controller,
        env_bundle=eval_env_bundle,
        config=config,
        eval_config=eval_config,
        run_id=run_id,
        progress_context=progress_context,
        ui_exports_root=ui_exports_root,
        artifacts_root=artifacts_root,
    )
    row = flatten_metrics_row(metrics_payload)

    manifest = {
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "config_path": str(resolve_path(config_path)),
        "eval_config_path": str(resolve_path(eval_config_path)),
        "schema_path": str(eval_env_bundle.schema_path),
        "dataset_name": eval_env_bundle.dataset_name,
        "seed": eval_env_bundle.seed,
        "step_count": step_count,
        "checkpoint_path": str(checkpoint_path),
        "training_curve_path": str(training_curve_path),
        "training_total_timesteps": int(config["training"]["total_timesteps"]),
        "split": config["env"]["split"],
        "control_mode": config["algorithm"]["control_mode"],
    }
    if artifact_id:
        manifest["artifact_id"] = artifact_id
        trained_on_split = checkpoint_payload.get("config", {}).get("env", {}).get("split")
        if trained_on_split is not None:
            manifest["trained_on_split"] = trained_on_split
        if label_mismatches:
            manifest["runtime_label_mismatches"] = {
                field: {"checkpoint": trained, "config": expected}
                for field, (trained, expected) in label_mismatches.items()
            }
    if job_id:
        manifest["job_id"] = job_id
    if job_dir is not None:
        manifest["job_dir"] = str(Path(job_dir))
    if ui_export_payload is not None:
        manifest["simulation_dir"] = ui_export_payload["simulation_dir"]
        manifest["playback_path"] = ui_export_payload["playback_path"]

    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics.json", metrics_payload)
    if ui_export_payload is not None:
        write_json(run_dir / "playback_manifest.json", ui_export_payload)
    if bool(config["evaluation"].get("save_rollout_trace", True)):
        write_json(run_dir / "rollout_trace.json", rollout_trace)

    write_csv_row(metrics_dir / f"{run_id}.csv", row)
    write_json(
        manifests_dir / "environment_lock.json",
        build_environment_lock(
            {
                "dataset_name": eval_env_bundle.dataset_name,
                "schema_path": str(eval_env_bundle.schema_path),
                "seed": eval_env_bundle.seed,
                "run_id": run_id,
            }
        ),
    )

    payload = {
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics.json"),
        "csv_path": str(metrics_dir / f"{run_id}.csv"),
        "run_id": run_id,
        "average_score": metrics_payload["average_score"],
        "checkpoint_path": str(checkpoint_path),
        "training_curve_path": str(training_curve_path),
    }

    if ui_export_payload is not None:
        payload["simulation_dir"] = _resolved_artifact_path(
            ui_export_payload["simulation_dir"],
            artifacts_root,
        )
        payload["playback_path"] = _resolved_artifact_path(
            ui_export_payload["playback_path"],
            artifacts_root,
        )

    if progress_context is not None:
        progress_context.artifact(
            kind="checkpoint",
            path=str(checkpoint_path),
            label="MAPPO checkpoint",
        )
        progress_context.artifact(
            kind="training_curve",
            path=str(training_curve_path),
            label="training curve",
        )
        if ui_export_payload is not None:
            progress_context.artifact(
                kind="playback",
                path=ui_export_payload["playback_path"],
                label="playback payload",
            )
            progress_context.artifact(
                kind="simulation_export",
                path=ui_export_payload["simulation_dir"],
                label="SimulationData export",
            )

    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/mappo/mappo_shared_ctde_reward_v2.yaml")
    parser.add_argument("--eval-config", default="configs/eval/default.yaml")
    args = parser.parse_args()
    print(json.dumps(run_mappo(config_path=args.config, eval_config_path=args.eval_config)))


if __name__ == "__main__":
    main()

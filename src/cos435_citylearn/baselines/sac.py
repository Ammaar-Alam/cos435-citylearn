from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import torch

from cos435_citylearn.algorithms.sac import (
    CentralizedSACController,
    SharedSACController,
    resolve_imported_checkpoint_path,
    resolve_reward_function,
    safe_load_checkpoint_payload,
    validate_checkpoint_env_compatibility,
    validate_checkpoint_runner_compatibility,
)
from cos435_citylearn.config import assert_training_allowed_on_split, load_yaml, resolve_path
from cos435_citylearn.env import (
    CentralizedEnvAdapter,
    PerBuildingEnvAdapter,
    make_citylearn_env,
)
from cos435_citylearn.eval import build_metrics_payload, flatten_metrics_row
from cos435_citylearn.io import ensure_parent, write_csv_row, write_json
from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.run_id import build_run_id
from cos435_citylearn.runtime import build_environment_lock, utc_now_iso
from cos435_citylearn.ui_exports import (
    DashboardCapture,
    build_live_preview_payload,
    export_simulation_bundle,
)


def _maybe_reset_reward(env: Any) -> None:
    reward_function = getattr(env, "reward_function", None)
    if reward_function is not None and hasattr(reward_function, "reset"):
        reward_function.reset()


def _resolved_artifact_path(path_value: str, artifacts_root: str | Path | None) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    root = RESULTS_DIR if artifacts_root is None else Path(artifacts_root)
    return str((root / path).resolve())


def _controller_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    total_timesteps = max(int(training.get("total_timesteps", 1)), 1)
    batch_size = int(training.get("batch_size", 256))
    default_standardize_start = min(max(batch_size * 4, 1024), total_timesteps)
    default_exploration_steps = min(max(batch_size * 2, 2048), total_timesteps)

    return {
        "hidden_dimension": list(training.get("hidden_dimension", [256, 256])),
        "discount": float(training.get("discount", 0.99)),
        "tau": float(training.get("tau", 5e-3)),
        "alpha": float(training.get("alpha", 0.2)),
        "lr": float(training.get("learning_rate", 3e-4)),
        "batch_size": batch_size,
        "replay_buffer_capacity": int(training.get("replay_buffer_size", 100_000)),
        "standardize_start_time_step": int(
            training.get("standardize_start_time_step", default_standardize_start)
        ),
        "end_exploration_time_step": int(
            training.get("exploration_steps", default_exploration_steps)
        ),
        "action_scaling_coefficienct": float(training.get("action_scaling_coefficient", 0.5)),
        "reward_scaling": float(training.get("reward_scaling", 5.0)),
        "update_per_time_step": int(training.get("update_per_time_step", 2)),
        "auto_entropy_tuning": bool(training.get("auto_entropy_tuning", True)),
    }


def _instantiate_controller(env: Any, config: dict[str, Any]):
    control_mode = config["algorithm"]["control_mode"]
    controller_kwargs = _controller_kwargs(config)

    if control_mode == "centralized":
        return CentralizedSACController(env, **controller_kwargs)
    if control_mode == "shared_dtde":
        return SharedSACController(
            env,
            shared_context_dimension=int(config["features"].get("shared_context_dimension", 4)),
            **controller_kwargs,
        )
    raise ValueError(f"unsupported SAC control mode: {control_mode}")


def _instantiate_controller_from_checkpoint(
    env: Any,
    checkpoint_payload: dict[str, Any],
):
    controller_state = checkpoint_payload["controller_state"]
    common_kwargs = {
        "hidden_dimension": list(controller_state["hidden_dimension"]),
        "discount": float(controller_state["discount"]),
        "tau": float(controller_state["tau"]),
        "alpha": float(controller_state["alpha"]),
        "lr": float(controller_state["lr"]),
        "batch_size": int(controller_state["batch_size"]),
        "replay_buffer_capacity": int(controller_state["replay_buffer_capacity"]),
        "standardize_start_time_step": int(controller_state["standardize_start_time_step"]),
        "end_exploration_time_step": int(controller_state["end_exploration_time_step"]),
        "action_scaling_coefficienct": float(controller_state["action_scaling_coefficient"]),
        "reward_scaling": float(controller_state["reward_scaling"]),
        "update_per_time_step": int(controller_state["update_per_time_step"]),
        "auto_entropy_tuning": bool(controller_state.get("auto_entropy_tuning", True)),
    }

    if controller_state["controller_type"] == "centralized_native":
        controller = CentralizedSACController(env, **common_kwargs)
    elif controller_state["controller_type"] == "shared_parameter_sac":
        controller = SharedSACController(
            env,
            shared_context_dimension=int(controller_state.get("shared_context_dimension", 4)),
            **common_kwargs,
        )
    else:
        controller_type = controller_state["controller_type"]
        raise ValueError(f"unknown SAC checkpoint controller type: {controller_type}")

    controller.reset()
    controller.load_checkpoint_state(controller_state)
    return controller


def _build_checkpoint_payload(
    *,
    run_id: str,
    config: dict[str, Any],
    env_bundle: Any,
    controller: Any,
    training_step: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "algorithm": config["algorithm"]["name"],
        "variant": config["algorithm"]["variant"],
        "control_mode": config["algorithm"]["control_mode"],
        "reward_version": config["reward"]["version"],
        "features_version": config["features"]["version"],
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


def _write_training_curve(path: Path, rows: list[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    fieldnames = list(rows[0]) if rows else ["step", "episode", "mean_reward"]

    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return target


def _load_imported_checkpoint(
    *,
    artifact_id: str,
    imported_artifacts_root: str | Path | None,
    artifacts_root: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    if imported_artifacts_root is None and artifacts_root is None:
        checkpoint_path = RESULTS_DIR / "runs" / artifact_id / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAC checkpoint not found for artifact: {checkpoint_path}")
    else:
        if imported_artifacts_root is None or artifacts_root is None:
            raise ValueError(
                "imported_artifacts_root and artifacts_root must both be set or both be None"
            )
        checkpoint_path = resolve_imported_checkpoint_path(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
    checkpoint_payload = safe_load_checkpoint_payload(checkpoint_path)
    return checkpoint_path, checkpoint_payload


def _build_adapter(env: Any):
    if env.central_agent:
        return CentralizedEnvAdapter(env)
    return PerBuildingEnvAdapter(env)


def _run_training_loop(
    *,
    controller: Any,
    env_bundle: Any,
    config: dict[str, Any],
    progress_context: Any | None,
) -> list[dict[str, Any]]:
    adapter = _build_adapter(env_bundle.env)
    observations = adapter.reset()
    _maybe_reset_reward(env_bundle.env)
    total_timesteps = int(config["training"]["total_timesteps"])
    preview_stride = max(1, int(config["evaluation"].get("preview_stride", 64)))
    log_interval_steps = max(1, int(config["training"].get("log_interval_steps", 128)))
    episode = 0
    step = 0
    episode_reward = 0.0
    curve_rows: list[dict[str, Any]] = []

    if progress_context is not None:
        progress_context.start(
            phase="training",
            total=total_timesteps,
            label=f"{config['algorithm']['variant']} training",
        )

    while step < total_timesteps:
        actions = controller.predict(observations, deterministic=False)
        applied_actions = adapter.clip_actions(actions)
        result = adapter.step(applied_actions)
        controller.update(
            observations,
            applied_actions,
            result.rewards,
            result.observations,
            done=result.terminated,
        )
        observations = result.observations
        episode_reward += float(sum(result.rewards) / max(len(result.rewards), 1))
        step += 1

        should_log = step == 1 or step == total_timesteps or step % log_interval_steps == 0
        if result.terminated:
            should_log = True

        if should_log:
            stats = controller.training_stats()
            curve_rows.append(
                {
                    "step": step,
                    "episode": episode,
                    "mean_reward": episode_reward,
                    "q1_loss": stats.get("q1_loss"),
                    "q2_loss": stats.get("q2_loss"),
                    "policy_loss": stats.get("policy_loss"),
                    "alpha": stats.get("alpha"),
                    "alpha_loss": stats.get("alpha_loss"),
                    "buffer_size": stats.get("buffer_size"),
                }
            )

        if progress_context is not None and (
            step == 1 or step == total_timesteps or step % preview_stride == 0
        ):
            progress_context.update(
                phase="training",
                current=step,
                total=total_timesteps,
                label=f"{config['algorithm']['variant']} training",
            )

        if result.terminated:
            observations = adapter.reset()
            _maybe_reset_reward(env_bundle.env)
            episode += 1
            episode_reward = 0.0

    return curve_rows


def _run_evaluation_loop(
    *,
    controller: Any,
    env_bundle: Any,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    run_id: str,
    progress_context: Any | None,
    ui_exports_root: str | Path | None,
    artifacts_root: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, int, list[dict[str, Any]]]:
    adapter = _build_adapter(env_bundle.env)
    observations = adapter.reset()
    _maybe_reset_reward(env_bundle.env)
    export_enabled = bool(eval_config["evaluation"].get("export_simulation_data", True))
    max_steps = eval_config["evaluation"].get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
        export_enabled = False

    capture = DashboardCapture(
        run_id=run_id,
        dataset_name=env_bundle.dataset_name,
        ui_exports_root=ui_exports_root,
        artifacts_root=artifacts_root,
        enabled=export_enabled,
        capture_frames=bool(eval_config["evaluation"].get("capture_render_frames", True)),
        max_frames=int(eval_config["evaluation"].get("max_render_frames", 60)),
        frame_width=int(eval_config["evaluation"].get("render_frame_width", 960)),
    )
    capture.configure(env_bundle.env)
    run_context = {
        "algorithm": config["algorithm"]["name"],
        "variant": config["algorithm"]["variant"],
        "split": config["env"]["split"],
        "seed": env_bundle.seed,
        "dataset_name": env_bundle.dataset_name,
        "run_id": run_id,
    }
    decision_total = max_steps or max(int(getattr(env_bundle.env, "time_steps", 0)) - 1, 0)
    preview_stride = max(1, int(config["evaluation"].get("preview_stride", 12)))
    trace_limit = int(config["evaluation"].get("trace_limit", 96))
    preview_trace: deque[dict[str, Any]] = deque(maxlen=trace_limit)
    rollout_trace: list[dict[str, Any]] = []
    playback_trace: list[dict[str, Any]] = [] if export_enabled else []
    step_index = 0

    if progress_context is not None:
        progress_context.update(
            phase="evaluation",
            current=0,
            total=decision_total,
            label=f"{config['algorithm']['variant']} evaluation",
            run_id=run_id,
        )

    while not adapter.done:
        if max_steps is not None and step_index >= max_steps:
            break

        actions = controller.predict(
            observations,
            deterministic=bool(eval_config["evaluation"].get("deterministic", True)),
        )
        applied_actions = adapter.clip_actions(actions)
        result = adapter.step(applied_actions)
        frame_payload = {
            "step": step_index,
            "actions": applied_actions,
            "rewards": result.rewards,
            "terminated": result.terminated,
        }
        preview_trace.append(frame_payload)
        if export_enabled:
            playback_trace.append(frame_payload)
        if bool(config["evaluation"].get("save_rollout_trace", True)) and step_index < trace_limit:
            rollout_trace.append(frame_payload)
        observations = result.observations
        capture.maybe_capture(env=env_bundle.env, step_index=step_index, force=result.terminated)
        step_index += 1

        if progress_context is not None and (
            step_index == 1 or step_index == decision_total or step_index % preview_stride == 0
        ):
            preview_payload = build_live_preview_payload(
                env=env_bundle.env,
                run_id=run_id,
                run_context=run_context,
                rollout_trace=list(preview_trace),
                capture=capture,
                current_step=max(step_index - 1, 0),
                history_limit=trace_limit,
                ui_exports_root=ui_exports_root,
                artifacts_root=artifacts_root,
            )
            progress_context.update(
                phase="evaluation",
                current=min(step_index, decision_total),
                total=decision_total,
                label=f"{config['algorithm']['variant']} evaluation",
                preview_payload=preview_payload,
                run_id=run_id,
            )

        if result.terminated:
            break

    metrics_payload = build_metrics_payload(env_bundle.env, run_context)
    ui_export_payload = None
    if export_enabled:
        ui_export_payload = export_simulation_bundle(
            env=env_bundle.env,
            run_id=run_id,
            run_context=run_context,
            metrics_payload=metrics_payload,
            rollout_trace=playback_trace,
            capture=capture,
            ui_exports_root=ui_exports_root,
            artifacts_root=artifacts_root,
        )

    return metrics_payload, ui_export_payload, step_index, rollout_trace


def run_sac(
    config_path: str | Path = "configs/train/sac/sac_central_baseline.yaml",
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
    allow_cross_reward_eval: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_config = load_yaml(eval_config_path)

    if split_override is not None:
        config["env"]["split"] = split_override
    if seed_override is not None:
        config["training"]["seed"] = int(seed_override)
    if lr_override is not None:
        config["training"]["learning_rate"] = float(lr_override)

    split_config_path = f"configs/splits/{config['env']['split']}.yaml"
    split_config = load_yaml(split_config_path)
    assert_training_allowed_on_split(split_config, artifact_id=artifact_id)

    reward_function = resolve_reward_function(config["reward"]["version"])
    central_agent = config["algorithm"]["control_mode"] == "centralized"

    env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=central_agent,
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
    # Collision tripwire: build_run_id should already guarantee uniqueness
    # (uuid / SLURM array+restart suffix), but if it ever regresses we want
    # FileExistsError here rather than two runs silently clobbering each
    # other's checkpoints / metrics / manifests.
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "checkpoint.pt"
    training_curve_path = run_dir / "training_curve.csv"

    label_mismatches: dict[str, tuple[Any, Any]] = {}
    if artifact_id is None:
        training_controller = _instantiate_controller(env_bundle.env, config)
        curve_rows = _run_training_loop(
            controller=training_controller,
            env_bundle=env_bundle,
            config=config,
            progress_context=progress_context,
        )
        checkpoint_payload = _build_checkpoint_payload(
            run_id=run_id,
            config=config,
            env_bundle=env_bundle,
            controller=training_controller,
            training_step=int(config["training"]["total_timesteps"]),
        )
        ensure_parent(checkpoint_path)
        torch.save(checkpoint_payload, checkpoint_path)
        _write_training_curve(training_curve_path, curve_rows)
    else:
        checkpoint_path, checkpoint_payload = _load_imported_checkpoint(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
        label_mismatches = validate_checkpoint_runner_compatibility(
            checkpoint_payload,
            config,
            allow_cross_reward_eval=allow_cross_reward_eval,
        )
        _write_training_curve(training_curve_path, [])
    if artifact_id is None:
        checkpoint_payload = safe_load_checkpoint_payload(checkpoint_path)

    eval_reward_function = resolve_reward_function(config["reward"]["version"])
    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=central_agent,
        reward_function=eval_reward_function,
    )
    validate_checkpoint_env_compatibility(
        checkpoint_payload,
        observation_names=eval_env_bundle.env.observation_names,
        action_names=eval_env_bundle.env.action_names,
    )
    evaluation_controller = _instantiate_controller_from_checkpoint(
        eval_env_bundle.env,
        checkpoint_payload,
    )
    metrics_payload, ui_export_payload, step_count, rollout_trace = _run_evaluation_loop(
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
            ui_export_payload["simulation_dir"], artifacts_root
        )
        payload["playback_path"] = _resolved_artifact_path(
            ui_export_payload["playback_path"], artifacts_root
        )

    if progress_context is not None:
        progress_context.artifact(
            kind="checkpoint",
            path=str(checkpoint_path),
            label="SAC checkpoint",
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
            gif_path = ui_export_payload.get("media", {}).get("gif_path")
            if gif_path:
                progress_context.artifact(
                    kind="gif",
                    path=gif_path,
                    label="render playback gif",
                )

    return payload

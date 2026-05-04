from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cos435_citylearn.algorithms._runtime_labels import (
    RUNTIME_LABEL_FIELDS,
    expected_runtime_labels,
    trained_runtime_labels,
)
from cos435_citylearn.algorithms.sac.checkpoints import resolve_imported_checkpoint_path
from cos435_citylearn.algorithms.sac.rewards import resolve_reward_function
from cos435_citylearn.algorithms.td3 import (
    SharedTD3Controller,
    safe_load_td3_checkpoint_payload,
    validate_td3_checkpoint_env_compatibility,
    validate_td3_checkpoint_runner_compatibility,
)
from cos435_citylearn.baselines.ppo import CityLearnGymWrapper
from cos435_citylearn.baselines.sac import (
    _resolved_artifact_path,
    _run_evaluation_loop,
    _run_training_loop,
    _write_training_curve,
)
from cos435_citylearn.config import assert_training_allowed_on_split, load_yaml, resolve_path
from cos435_citylearn.env import make_citylearn_env
from cos435_citylearn.eval import build_metrics_payload, flatten_metrics_row
from cos435_citylearn.io import ensure_parent, write_csv_row, write_json
from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.run_id import build_run_id
from cos435_citylearn.runtime import build_environment_lock, utc_now_iso

TRAINING_CURVE_FIELDS = ["step", "mean_reward", "actor_loss", "critic_loss"]
_CENTRAL_TD3_SIDECAR_NAME = "checkpoint_metadata.json"
_CENTRAL_TD3_SIDECAR_VERSION = 1


class TD3TrainingLogCallback(BaseCallback):
    def __init__(
        self,
        *,
        progress_context: Any | None,
        total_timesteps: int,
        variant: str,
        log_interval_steps: int,
    ) -> None:
        super().__init__()
        self.curve_rows: list[dict[str, Any]] = []
        self.progress_context = progress_context
        self.total_timesteps = int(total_timesteps)
        self.variant = variant
        self.log_interval_steps = max(1, int(log_interval_steps))

    def _on_step(self) -> bool:
        if self.num_timesteps == 1 or self.num_timesteps % self.log_interval_steps == 0:
            values = self.model.logger.name_to_value
            mean_reward = None
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            self.curve_rows.append(
                {
                    "step": int(self.num_timesteps),
                    "mean_reward": mean_reward,
                    "actor_loss": _maybe_float(values.get("train/actor_loss")),
                    "critic_loss": _maybe_float(values.get("train/critic_loss")),
                }
            )
            if self.progress_context is not None:
                self.progress_context.update(
                    phase="training",
                    current=int(self.num_timesteps),
                    total=self.total_timesteps,
                    label=f"{self.variant} training",
                )
        return True


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _build_central_td3_sidecar(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": _CENTRAL_TD3_SIDECAR_VERSION,
        "algorithm": config["algorithm"]["name"],
        "control_mode": config["algorithm"]["control_mode"],
        "variant": config["algorithm"].get("variant"),
        "reward_version": config.get("reward", {}).get("version"),
        "features_version": config.get("features", {}).get("version"),
        "config": config,
    }


def _write_central_td3_sidecar(run_dir: Path, config: dict[str, Any]) -> Path:
    sidecar_path = run_dir / _CENTRAL_TD3_SIDECAR_NAME
    write_json(sidecar_path, _build_central_td3_sidecar(config))
    return sidecar_path


def _load_central_td3_sidecar(model_path: Path) -> dict[str, Any] | None:
    sidecar_path = model_path.parent / _CENTRAL_TD3_SIDECAR_NAME
    if not sidecar_path.exists():
        return None
    return json.loads(sidecar_path.read_text())


def _validate_central_td3_sidecar(
    sidecar: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    allow_cross_reward_eval: bool,
    artifact_id: str,
) -> dict[str, tuple[Any, Any]]:
    if sidecar is None:
        if allow_cross_reward_eval:
            print(
                f"Warning: no {_CENTRAL_TD3_SIDECAR_NAME} for artifact '{artifact_id}'; "
                "allow_cross_reward_eval=True so skipping runtime-label check.",
                file=sys.stderr,
            )
            return {}
        raise ValueError(
            f"central TD3 artifact '{artifact_id}' is missing {_CENTRAL_TD3_SIDECAR_NAME}"
        )

    expected = expected_runtime_labels(config)
    trained = trained_runtime_labels(sidecar)
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in RUNTIME_LABEL_FIELDS:
        if trained.get(field) != expected.get(field):
            mismatches[field] = (trained.get(field), expected.get(field))
    if mismatches and not allow_cross_reward_eval:
        parts = [f"{k}: sidecar={t!r} config={e!r}" for k, (t, e) in mismatches.items()]
        raise ValueError(
            "central TD3 sidecar runtime metadata is incompatible with runner config; "
            "pass allow_cross_reward_eval=True to opt into cross-reward evaluation. "
            + "; ".join(parts)
        )
    return mismatches


def _resolve_central_td3_artifact_paths(
    artifact_id: str, artifacts_root: str | Path | None
) -> tuple[Path, Path]:
    root = Path(artifacts_root) if artifacts_root else RESULTS_DIR
    run_dir = root / "runs" / artifact_id
    model_path = run_dir / "td3_model.zip"
    vec_normalize_path = run_dir / "vec_normalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"TD3 model not found for artifact: {model_path}")
    if not vec_normalize_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found for artifact: {vec_normalize_path}")
    return model_path, vec_normalize_path


def _load_imported_central_td3_artifact(
    *,
    artifact_id: str,
    imported_artifacts_root: str | Path | None,
    artifacts_root: str | Path | None,
) -> tuple[Path, Path]:
    if imported_artifacts_root is None:
        return _resolve_central_td3_artifact_paths(artifact_id, artifacts_root)
    if artifacts_root is None:
        raise ValueError("artifacts_root must be set when imported_artifacts_root is provided")
    model_path = resolve_imported_checkpoint_path(
        artifact_id=artifact_id,
        imported_artifacts_root=imported_artifacts_root,
        artifacts_root=artifacts_root,
    )
    vec_normalize_path = model_path.parent / "vec_normalize.pkl"
    if not vec_normalize_path.exists():
        raise FileNotFoundError(
            "VecNormalize stats (vec_normalize.pkl) not found alongside imported "
            f"central TD3 model at {model_path.parent}."
        )
    return model_path, vec_normalize_path


def _load_central_td3_topology(model_path: Path, *, artifact_id: str) -> dict[str, Any]:
    topology_path = model_path.parent / "topology.json"
    if not topology_path.exists():
        raise FileNotFoundError(
            "Topology metadata (topology.json) not found alongside imported "
            f"central TD3 model for artifact '{artifact_id}' at {model_path.parent}."
        )
    return json.loads(topology_path.read_text())


def _validate_td3_topology(metadata: dict[str, Any], env: Any) -> None:
    env_obs_names = list(env.observation_names)
    env_act_names = list(env.action_names)
    ckpt_obs_names = metadata.get("observation_names", [])
    ckpt_act_names = metadata.get("action_names", [])
    if env_obs_names == ckpt_obs_names and env_act_names == ckpt_act_names:
        return
    ckpt_n = len(ckpt_obs_names)
    env_n = len(env_obs_names)
    if ckpt_n != env_n:
        raise ValueError(
            f"TD3 artifact was trained on {ckpt_n} buildings but target env has {env_n}. "
            "centralized TD3 cannot cross building counts; use td3_shared_dtde_*."
        )
    raise ValueError("TD3 artifact observation/action schema does not match target env")


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
        "policy_delay": int(training.get("policy_delay", 2)),
        "target_policy_noise": float(training.get("target_policy_noise", 0.2)),
        "target_noise_clip": float(training.get("target_noise_clip", 0.5)),
        "exploration_noise": float(training.get("exploration_noise", 0.1)),
    }


def _instantiate_shared_td3(env: Any, config: dict[str, Any]) -> SharedTD3Controller:
    return SharedTD3Controller(
        env,
        shared_context_dimension=int(config["features"].get("shared_context_dimension", 4)),
        **_controller_kwargs(config),
    )


def _instantiate_shared_td3_from_checkpoint(
    env: Any,
    checkpoint_payload: dict[str, Any],
) -> SharedTD3Controller:
    state = checkpoint_payload["controller_state"]
    controller = SharedTD3Controller(
        env,
        hidden_dimension=list(state["hidden_dimension"]),
        discount=float(state["discount"]),
        tau=float(state["tau"]),
        lr=float(state["lr"]),
        batch_size=int(state["batch_size"]),
        replay_buffer_capacity=int(state["replay_buffer_capacity"]),
        standardize_start_time_step=int(state["standardize_start_time_step"]),
        end_exploration_time_step=int(state["end_exploration_time_step"]),
        action_scaling_coefficienct=float(state["action_scaling_coefficient"]),
        reward_scaling=float(state["reward_scaling"]),
        update_per_time_step=int(state["update_per_time_step"]),
        shared_context_dimension=int(state["shared_context_dimension"]),
        policy_delay=int(state["policy_delay"]),
        target_policy_noise=float(state["target_policy_noise"]),
        target_noise_clip=float(state["target_noise_clip"]),
        exploration_noise=float(state["exploration_noise"]),
    )
    controller.reset()
    controller.load_checkpoint_state(state)
    return controller


def _build_shared_td3_checkpoint_payload(
    *,
    run_id: str,
    config: dict[str, Any],
    env_bundle: Any,
    controller: SharedTD3Controller,
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


def _load_imported_td3_checkpoint(
    *,
    artifact_id: str,
    imported_artifacts_root: str | Path | None,
    artifacts_root: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    if imported_artifacts_root is None:
        root = Path(artifacts_root) if artifacts_root is not None else RESULTS_DIR
        checkpoint_path = root / "runs" / artifact_id / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"shared TD3 checkpoint not found for artifact: {checkpoint_path}"
            )
    else:
        if artifacts_root is None:
            raise ValueError("artifacts_root must be set when imported_artifacts_root is provided")
        checkpoint_path = resolve_imported_checkpoint_path(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
    checkpoint_payload = safe_load_td3_checkpoint_payload(checkpoint_path)
    return checkpoint_path, checkpoint_payload


def _write_central_training_curve(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_central_td3(
    *,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    config_path: str | Path,
    eval_config_path: str | Path,
    output_root: str | Path | None,
    metrics_root: str | Path | None,
    manifests_root: str | Path | None,
    artifacts_root: str | Path | None,
    imported_artifacts_root: str | Path | None,
    artifact_id: str | None,
    job_id: str | None,
    job_dir: str | Path | None,
    progress_context: Any | None,
    allow_cross_reward_eval: bool,
) -> dict[str, Any]:
    reward_function = resolve_reward_function(config["reward"]["version"])
    seed = int(config["training"]["seed"])
    total_timesteps = int(config["training"]["total_timesteps"])
    variant = config["algorithm"].get("variant", "td3_central_baseline")
    run_id = build_run_id(
        algo="td3",
        variant=variant,
        split=config["env"]["split"],
        seed=seed,
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
    model_path = run_dir / "td3_model.zip"
    vec_normalize_path = run_dir / "vec_normalize.pkl"
    topology_path = run_dir / "topology.json"
    curve_path = run_dir / "training_curve.csv"
    label_mismatches: dict[str, tuple[Any, Any]] = {}

    if artifact_id is None:
        train_bundle = make_citylearn_env(
            config["env"]["base_config"],
            f"configs/splits/{config['env']['split']}.yaml",
            seed=seed,
            central_agent=True,
            reward_function=reward_function,
        )
        train_env = DummyVecEnv([lambda: CityLearnGymWrapper(train_bundle)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        action_dim = int(np.prod(train_env.action_space.shape))
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=float(config["training"].get("exploration_noise", 0.1)) * np.ones(action_dim),
        )
        model = TD3(
            "MlpPolicy",
            train_env,
            learning_rate=float(config["training"].get("learning_rate", 3e-4)),
            buffer_size=int(config["training"].get("replay_buffer_size", 100_000)),
            learning_starts=int(config["training"].get("learning_starts", 1000)),
            batch_size=int(config["training"].get("batch_size", 256)),
            tau=float(config["training"].get("tau", 0.005)),
            gamma=float(config["training"].get("discount", 0.99)),
            train_freq=(1, "step"),
            gradient_steps=int(config["training"].get("gradient_steps", 1)),
            action_noise=action_noise,
            policy_delay=int(config["training"].get("policy_delay", 2)),
            target_policy_noise=float(config["training"].get("target_policy_noise", 0.2)),
            target_noise_clip=float(config["training"].get("target_noise_clip", 0.5)),
            seed=seed,
            verbose=int(config["training"].get("sb3_verbose", 0)),
        )
        if progress_context is not None:
            progress_context.start(
                phase="training",
                total=total_timesteps,
                label=f"{variant} training",
            )
        callback = TD3TrainingLogCallback(
            progress_context=progress_context,
            total_timesteps=total_timesteps,
            variant=variant,
            log_interval_steps=int(config["training"].get("log_interval_steps", 1024)),
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        ensure_parent(model_path)
        model.save(str(model_path))
        train_env.save(str(vec_normalize_path))
        _write_central_td3_sidecar(run_dir, config)
        write_json(
            topology_path,
            {
                "observation_names": train_bundle.env.observation_names,
                "action_names": train_bundle.env.action_names,
                "observation_shape": list(train_env.observation_space.shape),
                "action_shape": list(train_env.action_space.shape),
                "trained_on_split": config["env"]["split"],
                "control_mode": "centralized",
            },
        )
        _write_central_training_curve(curve_path, callback.curve_rows)
        artifact_topology: dict[str, Any] | None = None
    else:
        imported_model_path, imported_vec_path = _load_imported_central_td3_artifact(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
        sidecar = _load_central_td3_sidecar(imported_model_path)
        label_mismatches = _validate_central_td3_sidecar(
            sidecar,
            config,
            allow_cross_reward_eval=allow_cross_reward_eval,
            artifact_id=artifact_id,
        )
        model = TD3.load(str(imported_model_path))
        ensure_parent(model_path)
        model.save(str(model_path))
        Path(vec_normalize_path).write_bytes(Path(imported_vec_path).read_bytes())
        if sidecar is not None:
            write_json(run_dir / _CENTRAL_TD3_SIDECAR_NAME, sidecar)
        artifact_topology = _load_central_td3_topology(
            imported_model_path,
            artifact_id=artifact_id,
        )
        write_json(topology_path, artifact_topology)
        _write_central_training_curve(curve_path, [])

    eval_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=seed,
        central_agent=True,
        reward_function=resolve_reward_function(config["reward"]["version"]),
    )
    if artifact_topology is not None:
        _validate_td3_topology(artifact_topology, eval_bundle.env)
    dummy_eval = DummyVecEnv([lambda: CityLearnGymWrapper(eval_bundle)])
    vn = VecNormalize.load(str(vec_normalize_path), dummy_eval)
    obs_mean = vn.obs_rms.mean
    obs_var = vn.obs_rms.var
    obs_eps = vn.epsilon
    clip_obs = vn.clip_obs
    raw_eval_env = CityLearnGymWrapper(eval_bundle)
    deterministic = bool(eval_config["evaluation"].get("deterministic", True))
    max_steps = eval_config["evaluation"].get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    trace_limit = int(config["evaluation"].get("trace_limit", 96))
    rollout_trace: list[dict[str, Any]] = []
    obs, _ = raw_eval_env.reset()
    done = False
    step_index = 0
    while not done:
        if max_steps is not None and step_index >= max_steps:
            break
        normalized = np.clip((obs - obs_mean) / np.sqrt(obs_var + obs_eps), -clip_obs, clip_obs)
        action, _ = model.predict(normalized, deterministic=deterministic)
        obs, _, terminated, truncated, info = raw_eval_env.step(action)
        if len(rollout_trace) < trace_limit:
            rollout_trace.append(
                {
                    "step": step_index,
                    "actions": info["per_building_actions"],
                    "rewards": info["per_building_rewards"],
                    "terminated": bool(terminated),
                }
            )
        step_index += 1
        done = bool(terminated or truncated)

    run_context = {
        "algorithm": "td3",
        "variant": variant,
        "split": config["env"]["split"],
        "seed": seed,
        "dataset_name": eval_bundle.dataset_name,
        "run_id": run_id,
    }
    metrics_payload = build_metrics_payload(eval_bundle.env, run_context)
    row = flatten_metrics_row(metrics_payload)
    manifest = {
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "config_path": str(resolve_path(config_path)),
        "eval_config_path": str(resolve_path(eval_config_path)),
        "dataset_name": eval_bundle.dataset_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "step_count": step_index,
        "model_path": str(model_path),
        "vec_normalize_path": str(vec_normalize_path),
        "training_curve_path": str(curve_path),
        "split": config["env"]["split"],
        "control_mode": "centralized",
    }
    if artifact_id:
        manifest["artifact_id"] = artifact_id
        if artifact_topology is not None and "trained_on_split" in artifact_topology:
            manifest["trained_on_split"] = artifact_topology["trained_on_split"]
        if label_mismatches:
            manifest["runtime_label_mismatches"] = {
                field: {"checkpoint": trained, "config": expected}
                for field, (trained, expected) in label_mismatches.items()
            }
    if job_id:
        manifest["job_id"] = job_id
    if job_dir is not None:
        manifest["job_dir"] = str(Path(job_dir))
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(run_dir / "rollout_trace.json", rollout_trace)
    write_csv_row(metrics_dir / f"{run_id}.csv", row)
    write_json(
        manifests_dir / "environment_lock.json",
        build_environment_lock(
            {
                "dataset_name": eval_bundle.dataset_name,
                "schema_path": str(eval_bundle.schema_path),
                "seed": seed,
                "run_id": run_id,
            }
        ),
    )
    return {
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics.json"),
        "csv_path": str(metrics_dir / f"{run_id}.csv"),
        "run_id": run_id,
        "average_score": metrics_payload["average_score"],
        "model_path": str(model_path),
        "vec_normalize_path": str(vec_normalize_path),
        "training_curve_path": str(curve_path),
    }


def _run_shared_td3(
    *,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    config_path: str | Path,
    eval_config_path: str | Path,
    output_root: str | Path | None,
    metrics_root: str | Path | None,
    manifests_root: str | Path | None,
    ui_exports_root: str | Path | None,
    artifacts_root: str | Path | None,
    imported_artifacts_root: str | Path | None,
    artifact_id: str | None,
    checkpoint_path: str | Path | None,
    job_id: str | None,
    job_dir: str | Path | None,
    progress_context: Any | None,
    allow_cross_reward_eval: bool,
) -> dict[str, Any]:
    env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=False,
        reward_function=resolve_reward_function(config["reward"]["version"]),
    )
    variant = config["algorithm"]["variant"]
    run_id = build_run_id(
        algo="td3",
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
    run_checkpoint_path = run_dir / "checkpoint.pt"
    training_curve_path = run_dir / "training_curve.csv"
    if artifact_id is not None and checkpoint_path is not None:
        raise ValueError("checkpoint evaluation accepts either artifact_id or checkpoint_path")

    label_mismatches: dict[str, tuple[Any, Any]] = {}
    if artifact_id is None and checkpoint_path is None:
        controller = _instantiate_shared_td3(env_bundle.env, config)
        curve_rows = _run_training_loop(
            controller=controller,
            env_bundle=env_bundle,
            config=config,
            progress_context=progress_context,
        )
        checkpoint_payload = _build_shared_td3_checkpoint_payload(
            run_id=run_id,
            config=config,
            env_bundle=env_bundle,
            controller=controller,
            training_step=int(config["training"]["total_timesteps"]),
        )
        ensure_parent(run_checkpoint_path)
        torch.save(checkpoint_payload, run_checkpoint_path)
        _write_training_curve(training_curve_path, curve_rows)
        resolved_checkpoint_path = run_checkpoint_path
    elif checkpoint_path is not None:
        resolved_checkpoint_path = resolve_path(checkpoint_path)
        checkpoint_payload = safe_load_td3_checkpoint_payload(resolved_checkpoint_path)
        label_mismatches = validate_td3_checkpoint_runner_compatibility(
            checkpoint_payload,
            config,
            allow_cross_reward_eval=allow_cross_reward_eval,
        )
        _write_training_curve(training_curve_path, [])
    else:
        resolved_checkpoint_path, checkpoint_payload = _load_imported_td3_checkpoint(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
        label_mismatches = validate_td3_checkpoint_runner_compatibility(
            checkpoint_payload,
            config,
            allow_cross_reward_eval=allow_cross_reward_eval,
        )
        _write_training_curve(training_curve_path, [])
    if artifact_id is None and checkpoint_path is None:
        checkpoint_payload = safe_load_td3_checkpoint_payload(resolved_checkpoint_path)

    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=False,
        reward_function=resolve_reward_function(config["reward"]["version"]),
    )
    validate_td3_checkpoint_env_compatibility(
        checkpoint_payload,
        observation_names=eval_env_bundle.env.observation_names,
        action_names=eval_env_bundle.env.action_names,
    )
    evaluation_controller = _instantiate_shared_td3_from_checkpoint(
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
        "checkpoint_path": str(resolved_checkpoint_path),
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
        "checkpoint_path": str(resolved_checkpoint_path),
        "training_curve_path": str(training_curve_path),
    }
    if ui_export_payload is not None:
        payload["simulation_dir"] = _resolved_artifact_path(
            ui_export_payload["simulation_dir"], artifacts_root
        )
        payload["playback_path"] = _resolved_artifact_path(
            ui_export_payload["playback_path"], artifacts_root
        )
    return payload


def run_td3(
    config_path: str | Path = "configs/train/td3/td3_central_baseline.yaml",
    eval_config_path: str | Path = "configs/eval/default.yaml",
    output_root: str | Path | None = None,
    metrics_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    ui_exports_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    imported_artifacts_root: str | Path | None = None,
    artifact_id: str | None = None,
    checkpoint_path: str | Path | None = None,
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
    split_config = load_yaml(f"configs/splits/{config['env']['split']}.yaml")
    assert_training_allowed_on_split(
        split_config,
        artifact_id=artifact_id,
        checkpoint_path=checkpoint_path,
    )
    if config["algorithm"]["control_mode"] == "centralized":
        if checkpoint_path is not None:
            raise ValueError("central TD3 accepts artifact_id, not checkpoint_path")
        return _run_central_td3(
            config=config,
            eval_config=eval_config,
            config_path=config_path,
            eval_config_path=eval_config_path,
            output_root=output_root,
            metrics_root=metrics_root,
            manifests_root=manifests_root,
            artifacts_root=artifacts_root,
            imported_artifacts_root=imported_artifacts_root,
            artifact_id=artifact_id,
            job_id=job_id,
            job_dir=job_dir,
            progress_context=progress_context,
            allow_cross_reward_eval=allow_cross_reward_eval,
        )
    if config["algorithm"]["control_mode"] == "shared_dtde":
        return _run_shared_td3(
            config=config,
            eval_config=eval_config,
            config_path=config_path,
            eval_config_path=eval_config_path,
            output_root=output_root,
            metrics_root=metrics_root,
            manifests_root=manifests_root,
            ui_exports_root=ui_exports_root,
            artifacts_root=artifacts_root,
            imported_artifacts_root=imported_artifacts_root,
            artifact_id=artifact_id,
            checkpoint_path=checkpoint_path,
            job_id=job_id,
            job_dir=job_dir,
            progress_context=progress_context,
            allow_cross_reward_eval=allow_cross_reward_eval,
        )
    raise ValueError(f"unsupported TD3 control mode: {config['algorithm']['control_mode']}")

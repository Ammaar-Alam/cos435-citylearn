# PPO baseline: centralized path uses SB3, shared_dtde path uses our SharedPPOController
from __future__ import annotations

import csv
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cos435_citylearn.algorithms.ppo import (
    SharedPPOController,
    safe_load_ppo_checkpoint_payload,
    validate_ppo_checkpoint_env_compatibility,
    validate_ppo_checkpoint_runner_compatibility,
)
from cos435_citylearn.algorithms.sac.rewards import resolve_reward_function
from cos435_citylearn.config import assert_training_allowed_on_split, load_yaml, resolve_path
from cos435_citylearn.env import (
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

TRAINING_CURVE_FIELDS = [
    "step",
    "mean_reward",
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
]


class CityLearnGymWrapper(gym.Env):
    # flatten per-building obs/actions for SB3 and stash the raw per-building stuff in info
    def __init__(self, env_bundle):
        super().__init__()
        self.citylearn_env = env_bundle.env

        obs_dim = sum(box.shape[0] for box in self.citylearn_env.observation_space)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        # use actual bounds from CityLearn action spaces
        act_lows = np.concatenate([box.low for box in self.citylearn_env.action_space])
        act_highs = np.concatenate([box.high for box in self.citylearn_env.action_space])
        self.action_space = spaces.Box(
            low=act_lows.astype(np.float32),
            high=act_highs.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs = self.citylearn_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        # reset stateful reward funcs manually
        reward_function = getattr(self.citylearn_env, "reward_function", None)
        if reward_function is not None and hasattr(reward_function, "reset"):
            reward_function.reset()
        flat = np.concatenate([np.array(o, dtype=np.float32) for o in obs])
        return flat, {}

    def step(self, action):
        # split the flat action back into per-building chunks
        per_building_actions = []
        idx = 0
        for box in self.citylearn_env.action_space:
            dim = box.shape[0]
            a = np.clip(action[idx : idx + dim], box.low, box.high)
            per_building_actions.append(a.astype(float).tolist())
            idx += dim

        result = self.citylearn_env.step(per_building_actions)

        if len(result) == 4:
            obs, rewards, terminated, info = result
            truncated = False
        else:
            obs, rewards, terminated, truncated, info = result

        flat_obs = np.concatenate([np.array(o, dtype=np.float32) for o in obs])
        per_building_rewards = [float(r) for r in rewards]
        total_reward = float(sum(per_building_rewards)) / max(len(per_building_rewards), 1)

        # stash per-building details for the trace writer
        step_info = dict(info) if info else {}
        step_info["per_building_actions"] = per_building_actions
        step_info["per_building_rewards"] = per_building_rewards

        return flat_obs, total_reward, bool(terminated), bool(truncated), step_info


class TrainingLogCallback(BaseCallback):
    # grab SB3's per-iter stats so we can write a training_curve.csv
    def __init__(self, progress_context=None, total_timesteps=0, variant=""):
        super().__init__()
        self.curve_rows = []
        self.progress_context = progress_context
        self.total_timesteps = total_timesteps
        self.variant = variant

    def _on_rollout_end(self):
        # log once per rollout, after SB3 has updated its logger with train stats
        values = self.model.logger.name_to_value

        def _get(key):
            raw = values.get(key)
            return None if raw is None else float(raw)

        mean_reward = None
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))

        self.curve_rows.append({
            "step": int(self.num_timesteps),
            "mean_reward": mean_reward,
            "policy_loss": _get("train/policy_gradient_loss"),
            "value_loss": _get("train/value_loss"),
            "entropy_loss": _get("train/entropy_loss"),
            "approx_kl": _get("train/approx_kl"),
            "clip_fraction": _get("train/clip_fraction"),
            "explained_variance": _get("train/explained_variance"),
        })

        if self.progress_context is not None and self.total_timesteps > 0:
            self.progress_context.update(
                phase="training",
                current=int(self.num_timesteps),
                total=int(self.total_timesteps),
                label=f"{self.variant} training",
            )

    def _on_step(self):
        return True


def _resolve_artifact_paths(
    artifact_id: str, artifacts_root: str | Path | None
) -> tuple[Path, Path]:
    root = Path(artifacts_root) if artifacts_root else RESULTS_DIR
    run_dir = root / "runs" / artifact_id

    model_path = run_dir / "ppo_model.zip"
    vec_normalize_path = run_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found for artifact: {model_path}")
    if not vec_normalize_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found for artifact: {vec_normalize_path}")

    return model_path, vec_normalize_path


def _validate_ppo_topology(metadata: dict[str, Any], env: Any) -> None:
    # central PPO bakes building count into the policy, so 3 -> 6 would crash opaquely
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
            f"PPO artifact was trained on {ckpt_n} buildings but target env has {env_n}. "
            "centralized PPO cannot cross building counts; retrain on the target split "
            "(or use sac_shared_dtde_* which is topology-invariant)."
        )
    raise ValueError(
        "PPO artifact observation/action schema does not match target env; "
        "the two datasets expose different building features."
    )


def run_ppo(
    config_path: str | Path = "configs/train/ppo/ppo_central_baseline.yaml",
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

    split_config_path = f"configs/splits/{config['env']['split']}.yaml"
    split_config = load_yaml(split_config_path)
    assert_training_allowed_on_split(split_config, artifact_id=artifact_id)

    if config["algorithm"].get("control_mode") == "shared_dtde":
        return _run_shared_ppo(
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
            job_id=job_id,
            job_dir=job_dir,
            progress_context=progress_context,
        )

    reward_version = config["reward"]["version"]
    reward_function = resolve_reward_function(reward_version)

    seed = int(config["training"]["seed"])
    total_timesteps = int(config["training"]["total_timesteps"])
    variant = config["algorithm"].get("variant", "ppo_central_baseline")
    trace_limit = int(config["evaluation"].get("trace_limit", 96))
    save_rollout_trace = bool(config["evaluation"].get("save_rollout_trace", True))

    run_id = build_run_id(
        algo="ppo",
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
    model_path = run_dir / "ppo_model.zip"
    vec_normalize_path = run_dir / "vec_normalize.pkl"
    topology_path = run_dir / "topology.json"
    curve_path = run_dir / "training_curve.csv"

    if artifact_id is None:
        # --- train from scratch ---
        train_bundle = make_citylearn_env(
            config["env"]["base_config"],
            f"configs/splits/{config['env']['split']}.yaml",
            seed=seed,
            central_agent=True,
            reward_function=reward_function,
        )
        train_env = DummyVecEnv([lambda: CityLearnGymWrapper(train_bundle)])
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        lr = float(config["training"].get("learning_rate", 3e-4))
        rollout_steps = int(config["training"].get("rollout_steps", 2048))

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
            n_steps=rollout_steps,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            seed=seed,
            verbose=1,
        )

        if progress_context is not None:
            progress_context.start(
                phase="training",
                total=total_timesteps,
                label=f"{variant} training",
            )

        print(f"Training PPO for {total_timesteps} timesteps...", file=sys.stderr)
        log_callback = TrainingLogCallback(
            progress_context=progress_context,
            total_timesteps=total_timesteps,
            variant=variant,
        )
        model.learn(total_timesteps=total_timesteps, callback=log_callback)

        # callback logs last iter's train stats, so append one final row after learn() returns
        final_values = model.logger.name_to_value

        def _final(key):
            raw = final_values.get(key)
            return None if raw is None else float(raw)

        final_mean_reward = None
        if len(model.ep_info_buffer) > 0:
            final_mean_reward = float(np.mean([ep["r"] for ep in model.ep_info_buffer]))

        log_callback.curve_rows.append({
            "step": int(total_timesteps),
            "mean_reward": final_mean_reward,
            "policy_loss": _final("train/policy_gradient_loss"),
            "value_loss": _final("train/value_loss"),
            "entropy_loss": _final("train/entropy_loss"),
            "approx_kl": _final("train/approx_kl"),
            "clip_fraction": _final("train/clip_fraction"),
            "explained_variance": _final("train/explained_variance"),
        })

        # save model and VecNormalize stats
        ensure_parent(model_path)
        model.save(str(model_path))
        train_env.save(str(vec_normalize_path))

        # save topology for cross-topology preflight
        write_json(topology_path, {
            "observation_names": train_bundle.env.observation_names,
            "action_names": train_bundle.env.action_names,
            "observation_shape": list(train_env.observation_space.shape),
            "action_shape": list(train_env.action_space.shape),
            "trained_on_split": config["env"]["split"],
            "control_mode": "centralized",
        })

        # write training curve
        ensure_parent(curve_path)
        with curve_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(log_callback.curve_rows)

        artifact_topology: dict[str, Any] | None = None
    else:
        # --- load saved artifact, skip training ---
        resolved_root = imported_artifacts_root or artifacts_root
        imported_model_path, imported_vec_path = _resolve_artifact_paths(
            artifact_id, resolved_root
        )
        print(f"Loading PPO artifact '{artifact_id}' from {imported_model_path}", file=sys.stderr)
        model = PPO.load(str(imported_model_path))

        # copy artifacts into this run's dir so it's self-contained
        ensure_parent(model_path)
        model.save(str(model_path))
        Path(vec_normalize_path).write_bytes(Path(imported_vec_path).read_bytes())

        imported_topology_path = Path(imported_model_path).parent / "topology.json"
        artifact_topology = None
        if imported_topology_path.exists():
            artifact_topology = json.loads(imported_topology_path.read_text())
            write_json(topology_path, artifact_topology)
        else:
            print(
                f"Warning: no topology.json for artifact '{artifact_id}'; "
                "skipping PPO preflight check",
                file=sys.stderr,
            )

        # write empty training curve for consistency
        ensure_parent(curve_path)
        with curve_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_CURVE_FIELDS)
            writer.writeheader()

    # --- evaluate on a fresh env with saved normalization stats ---
    print("Evaluating trained PPO agent...", file=sys.stderr)
    eval_reward_function = resolve_reward_function(reward_version)
    eval_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=seed,
        central_agent=True,
        reward_function=eval_reward_function,
    )
    if artifact_topology is not None:
        _validate_ppo_topology(artifact_topology, eval_bundle.env)
    # apply VecNormalize stats manually so the env doesn't auto-reset and drop the last episode
    dummy_eval = DummyVecEnv([lambda: CityLearnGymWrapper(eval_bundle)])
    vn = VecNormalize.load(str(vec_normalize_path), dummy_eval)
    obs_mean = vn.obs_rms.mean
    obs_var = vn.obs_rms.var
    obs_eps = vn.epsilon
    clip_obs = vn.clip_obs

    raw_eval_env = CityLearnGymWrapper(eval_bundle)
    deterministic = bool(eval_config["evaluation"].get("deterministic", True))
    max_steps = eval_config["evaluation"].get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)

    rollout_trace: list[dict[str, Any]] = []
    obs, _ = raw_eval_env.reset()
    done = False
    step_index = 0
    decision_total = max_steps or max(int(getattr(eval_bundle.env, "time_steps", 0)) - 1, 0)

    if progress_context is not None:
        progress_context.update(
            phase="evaluation",
            current=0,
            total=decision_total,
            label=f"{variant} evaluation",
            run_id=run_id,
        )

    while not done:
        if max_steps is not None and step_index >= max_steps:
            break

        normalized = np.clip((obs - obs_mean) / np.sqrt(obs_var + obs_eps), -clip_obs, clip_obs)
        action, _ = model.predict(normalized, deterministic=deterministic)
        obs, _, terminated, truncated, info = raw_eval_env.step(action)

        if save_rollout_trace and len(rollout_trace) < trace_limit:
            rollout_trace.append({
                "step": step_index,
                "actions": info["per_building_actions"],
                "rewards": info["per_building_rewards"],
                "terminated": bool(terminated),
            })

        step_index += 1
        done = terminated or truncated

        if progress_context is not None and (
            step_index == 1 or step_index == decision_total or step_index % 12 == 0
        ):
            progress_context.update(
                phase="evaluation",
                current=min(step_index, decision_total),
                total=decision_total,
                label=f"{variant} evaluation",
                run_id=run_id,
            )

    # --- compute metrics via shared pipeline ---
    run_context = {
        "algorithm": "ppo",
        "variant": variant,
        "split": config["env"]["split"],
        "seed": seed,
        "dataset_name": eval_bundle.dataset_name,
        "run_id": run_id,
    }
    metrics_payload = build_metrics_payload(eval_bundle.env, run_context)
    row = flatten_metrics_row(metrics_payload)

    # --- save everything ---
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
    }
    if artifact_id:
        manifest["artifact_id"] = artifact_id
        if artifact_topology is not None and "trained_on_split" in artifact_topology:
            manifest["trained_on_split"] = artifact_topology["trained_on_split"]
    if job_id:
        manifest["job_id"] = job_id
    if job_dir is not None:
        manifest["job_dir"] = str(Path(job_dir))

    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics.json", metrics_payload)
    if save_rollout_trace:
        write_json(run_dir / "rollout_trace.json", rollout_trace)
    write_csv_row(metrics_dir / f"{run_id}.csv", row)
    write_json(
        manifests_dir / "environment_lock.json",
        build_environment_lock({
            "dataset_name": eval_bundle.dataset_name,
            "schema_path": str(eval_bundle.schema_path),
            "seed": seed,
            "run_id": run_id,
        }),
    )

    if progress_context is not None:
        progress_context.artifact(
            kind="model",
            path=str(model_path),
            label="PPO model",
        )
        progress_context.artifact(
            kind="vec_normalize",
            path=str(vec_normalize_path),
            label="VecNormalize stats",
        )
        progress_context.artifact(
            kind="training_curve",
            path=str(curve_path),
            label="training curve",
        )

    print(f"Average score: {metrics_payload['average_score']}", file=sys.stderr)

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


SHARED_PPO_TRAINING_CURVE_FIELDS = [
    "step",
    "iteration",
    "mean_reward",
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
    "n_updates",
    "total_updates",
]


def _shared_ppo_controller_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    features = config.get("features", {})
    return {
        "hidden_dimension": list(training.get("hidden_dimension", [64, 64])),
        "lr": float(training.get("learning_rate", 3e-4)),
        "clip_range": float(training.get("clip_range", 0.2)),
        "n_epochs": int(training.get("n_epochs", 10)),
        "minibatch_size": int(training.get("minibatch_size", 64)),
        "gamma": float(training.get("gamma", 0.99)),
        "gae_lambda": float(training.get("gae_lambda", 0.95)),
        "ent_coef": float(training.get("ent_coef", 0.01)),
        "vf_coef": float(training.get("vf_coef", 0.5)),
        "max_grad_norm": float(training.get("max_grad_norm", 0.5)),
        "rollout_steps": int(training.get("rollout_steps", 2048)),
        "reward_scaling": float(training.get("reward_scaling", 1.0)),
        "shared_context_dimension": int(features.get("shared_context_dimension", 4)),
        "shared_context_version": str(features.get("shared_context_version", "v2")),
        "normalize_observations": bool(training.get("normalize_observations", True)),
        "normalize_rewards": bool(training.get("normalize_rewards", True)),
        "target_kl": (
            None if training.get("target_kl") is None else float(training["target_kl"])
        ),
    }


def _build_shared_ppo_checkpoint_payload(
    *,
    run_id: str,
    config: dict[str, Any],
    env_bundle: Any,
    controller: SharedPPOController,
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


def _instantiate_shared_ppo_from_checkpoint(
    env: Any,
    checkpoint_payload: dict[str, Any],
) -> SharedPPOController:
    controller_state = checkpoint_payload["controller_state"]
    controller = SharedPPOController(
        env,
        hidden_dimension=list(controller_state["hidden_dimension"]),
        lr=float(controller_state["lr"]),
        clip_range=float(controller_state["clip_range"]),
        n_epochs=int(controller_state["n_epochs"]),
        minibatch_size=int(controller_state["minibatch_size"]),
        gamma=float(controller_state["gamma"]),
        gae_lambda=float(controller_state["gae_lambda"]),
        ent_coef=float(controller_state["ent_coef"]),
        vf_coef=float(controller_state["vf_coef"]),
        max_grad_norm=float(controller_state["max_grad_norm"]),
        rollout_steps=int(controller_state["rollout_steps"]),
        reward_scaling=float(controller_state["reward_scaling"]),
        shared_context_dimension=int(controller_state["shared_context_dimension"]),
        shared_context_version=str(controller_state["shared_context_version"]),
        normalize_observations=bool(controller_state.get("normalize_observations", True)),
        normalize_rewards=bool(controller_state.get("normalize_rewards", True)),
        target_kl=(
            None
            if controller_state.get("target_kl") is None
            else float(controller_state["target_kl"])
        ),
    )
    controller.load_checkpoint_state(controller_state)
    return controller


def _load_imported_ppo_checkpoint(
    *,
    artifact_id: str,
    imported_artifacts_root: str | Path | None,
    artifacts_root: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    if imported_artifacts_root is None:
        # re-eval a run we already trained: look under artifacts_root (or default results)
        root = Path(artifacts_root) if artifacts_root is not None else RESULTS_DIR
        checkpoint_path = root / "runs" / artifact_id / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"shared PPO checkpoint not found for artifact: {checkpoint_path}"
            )
    else:
        if artifacts_root is None:
            raise ValueError(
                "artifacts_root must be set when imported_artifacts_root is provided"
            )
        from cos435_citylearn.algorithms.sac.checkpoints import resolve_imported_checkpoint_path
        checkpoint_path = resolve_imported_checkpoint_path(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
    checkpoint_payload = safe_load_ppo_checkpoint_payload(checkpoint_path)
    return checkpoint_path, checkpoint_payload


def _run_shared_ppo_training_loop(
    *,
    controller: SharedPPOController,
    env_bundle: Any,
    config: dict[str, Any],
    progress_context: Any | None,
) -> list[dict[str, Any]]:
    adapter = PerBuildingEnvAdapter(env_bundle.env)
    observations = adapter.reset()
    reward_function = getattr(env_bundle.env, "reward_function", None)
    if reward_function is not None and hasattr(reward_function, "reset"):
        reward_function.reset()

    total_timesteps = int(config["training"]["total_timesteps"])
    rollout_steps = int(config["training"]["rollout_steps"])
    variant = config["algorithm"]["variant"]

    if progress_context is not None:
        progress_context.start(
            phase="training",
            total=total_timesteps,
            label=f"{variant} training",
        )

    curve_rows: list[dict[str, Any]] = []
    step = 0
    iteration = 0
    episode_reward = 0.0

    while step < total_timesteps:
        steps_this_rollout = 0
        controller.rollout_buffer.reset()
        while steps_this_rollout < rollout_steps and step < total_timesteps:
            step_payload = controller.sample_rollout_step(observations)
            applied_actions = adapter.clip_actions(step_payload["actions_list"])
            result = adapter.step(applied_actions)
            controller.store_rollout_step(
                step_payload=step_payload,
                rewards=result.rewards,
                done=result.terminated,
            )
            episode_reward += float(sum(result.rewards) / max(len(result.rewards), 1))
            step += 1
            steps_this_rollout += 1
            observations = result.observations

            if result.terminated:
                observations = adapter.reset()
                if reward_function is not None and hasattr(reward_function, "reset"):
                    reward_function.reset()

            progress_interval = max(1, rollout_steps // 4)
            if progress_context is not None and (
                step == total_timesteps or step % progress_interval == 0
            ):
                progress_context.update(
                    phase="training",
                    current=step,
                    total=total_timesteps,
                    label=f"{variant} training",
                )

        if controller.rollout_buffer.size == 0:
            break

        stats = controller.finish_rollout(observations)
        iteration += 1
        curve_rows.append(
            {
                "step": step,
                "iteration": iteration,
                "mean_reward": episode_reward,
                "policy_loss": stats.get("policy_loss"),
                "value_loss": stats.get("value_loss"),
                "entropy_loss": stats.get("entropy_loss"),
                "approx_kl": stats.get("approx_kl"),
                "clip_fraction": stats.get("clip_fraction"),
                "n_updates": stats.get("n_updates"),
                "total_updates": stats.get("total_updates"),
            }
        )
        episode_reward = 0.0

    return curve_rows


def _write_shared_ppo_training_curve(path: Path, rows: list[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHARED_PPO_TRAINING_CURVE_FIELDS)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return target


def _run_shared_ppo_evaluation_loop(
    *,
    controller: SharedPPOController,
    env_bundle: Any,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    run_id: str,
    progress_context: Any | None,
    ui_exports_root: str | Path | None,
    artifacts_root: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, int, list[dict[str, Any]]]:
    adapter = PerBuildingEnvAdapter(env_bundle.env)
    observations = adapter.reset()
    reward_function = getattr(env_bundle.env, "reward_function", None)
    if reward_function is not None and hasattr(reward_function, "reset"):
        reward_function.reset()

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


def _resolved_artifact_path(path_value: str, artifacts_root: str | Path | None) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    root = RESULTS_DIR if artifacts_root is None else Path(artifacts_root)
    return str((root / path).resolve())


def _run_shared_ppo(
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
    job_id: str | None,
    job_dir: str | Path | None,
    progress_context: Any | None,
) -> dict[str, Any]:
    reward_function = resolve_reward_function(config["reward"]["version"])
    env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
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
    checkpoint_path = run_dir / "checkpoint.pt"
    training_curve_path = run_dir / "training_curve.csv"

    if artifact_id is None:
        training_controller = SharedPPOController(
            env_bundle.env,
            **_shared_ppo_controller_kwargs(config),
        )
        curve_rows = _run_shared_ppo_training_loop(
            controller=training_controller,
            env_bundle=env_bundle,
            config=config,
            progress_context=progress_context,
        )
        checkpoint_payload = _build_shared_ppo_checkpoint_payload(
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
        checkpoint_path, checkpoint_payload = _load_imported_ppo_checkpoint(
            artifact_id=artifact_id,
            imported_artifacts_root=imported_artifacts_root,
            artifacts_root=artifacts_root,
        )
        validate_ppo_checkpoint_runner_compatibility(checkpoint_payload, config)
        _write_shared_ppo_training_curve(training_curve_path, [])

    if artifact_id is None:
        checkpoint_payload = safe_load_ppo_checkpoint_payload(checkpoint_path)

    eval_reward_function = resolve_reward_function(config["reward"]["version"])
    eval_env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=False,
        reward_function=eval_reward_function,
    )
    validate_ppo_checkpoint_env_compatibility(
        checkpoint_payload,
        observation_names=eval_env_bundle.env.observation_names,
        action_names=eval_env_bundle.env.action_names,
    )
    evaluation_controller = _instantiate_shared_ppo_from_checkpoint(
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
            label="shared PPO checkpoint",
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

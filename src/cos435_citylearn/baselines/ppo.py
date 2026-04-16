"""
PPO baseline for CityLearn using Stable-Baselines3.

Wraps the CityLearn environment in a standard Gymnasium interface so SB3's PPO
can use it directly. Uses VecNormalize for observation/reward normalization,
which is standard practice for PPO on continuous control tasks.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cos435_citylearn.algorithms.sac.rewards import resolve_reward_function
from cos435_citylearn.config import load_yaml, resolve_path
from cos435_citylearn.env import make_citylearn_env
from cos435_citylearn.eval import build_metrics_payload, flatten_metrics_row
from cos435_citylearn.io import ensure_parent, write_csv_row, write_json
from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.run_id import build_run_id
from cos435_citylearn.runtime import build_environment_lock, utc_now_iso


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
    """Wraps CityLearn in a standard Gymnasium env for centralized control.

    Flattens list-of-lists observations and actions into single 1D arrays.
    Stashes per-building rewards and actions in the info dict so the eval
    loop can save a rollout trace that matches SAC's format.
    """

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
        # stateful reward functions (reward_v1/v2/v3) need explicit reset — CityLearnEnv.reset() does not call reward_function.reset()
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
    """Logs PPO training stats to a list for writing a training curve CSV.

    Pulls per-iteration stats from SB3's logger (populated after each rollout).
    """

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


def _resolve_artifact_paths(artifact_id: str, artifacts_root: str | Path | None) -> tuple[Path, Path]:
    """Resolve model + VecNormalize paths for a previously-saved PPO run.

    Treats artifact_id as a run_id and looks under results/runs/<artifact_id>/.
    Matches the layout that run_ppo writes on training completion.
    """
    root = Path(artifacts_root) if artifacts_root else RESULTS_DIR
    run_dir = root / "runs" / artifact_id

    model_path = run_dir / "ppo_model.zip"
    vec_normalize_path = run_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found for artifact: {model_path}")
    if not vec_normalize_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found for artifact: {vec_normalize_path}")

    return model_path, vec_normalize_path


def run_ppo(
    config_path: str | Path = "configs/train/ppo/ppo_central_baseline.yaml",
    eval_config_path: str | Path = "configs/eval/default.yaml",
    output_root: str | Path | None = None,
    metrics_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    imported_artifacts_root: str | Path | None = None,
    artifact_id: str | None = None,
    job_id: str | None = None,
    job_dir: str | Path | None = None,
    progress_context: Any | None = None,
    **kwargs,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_config = load_yaml(eval_config_path)

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

        print(f"Training PPO for {total_timesteps} timesteps...")
        log_callback = TrainingLogCallback(
            progress_context=progress_context,
            total_timesteps=total_timesteps,
            variant=variant,
        )
        model.learn(total_timesteps=total_timesteps, callback=log_callback)

        # _on_rollout_end fires before PPO.train(), so the callback captures train/* stats
        # from the *previous* iteration. Append one final row with the last iteration's
        # stats (which are now in the logger after learn() returns).
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

        # write training curve
        ensure_parent(curve_path)
        with curve_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(log_callback.curve_rows)
    else:
        # --- load saved artifact, skip training ---
        resolved_root = imported_artifacts_root or artifacts_root
        imported_model_path, imported_vec_path = _resolve_artifact_paths(
            artifact_id, resolved_root
        )
        print(f"Loading PPO artifact '{artifact_id}' from {imported_model_path}")
        model = PPO.load(str(imported_model_path))

        # copy artifacts into this run's dir so it's self-contained
        ensure_parent(model_path)
        model.save(str(model_path))
        Path(vec_normalize_path).write_bytes(Path(imported_vec_path).read_bytes())

        # write empty training curve for consistency
        ensure_parent(curve_path)
        with curve_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_CURVE_FIELDS)
            writer.writeheader()

    # --- evaluate on a fresh env with saved normalization stats ---
    print("Evaluating trained PPO agent...")
    eval_reward_function = resolve_reward_function(reward_version)
    eval_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=seed,
        central_agent=True,
        reward_function=eval_reward_function,
    )
    # load saved VecNormalize stats for manual normalization
    # (using the vec env directly would auto-reset on done and wipe the final episode state)
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
    }
    if artifact_id:
        manifest["artifact_id"] = artifact_id
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

    print(f"Average score: {metrics_payload['average_score']}")

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

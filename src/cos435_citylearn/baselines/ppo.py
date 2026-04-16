from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from cos435_citylearn.algorithms.sac.rewards import resolve_reward_function
from cos435_citylearn.config import load_yaml, resolve_path
from cos435_citylearn.env import make_citylearn_env
from cos435_citylearn.eval import build_metrics_payload, flatten_metrics_row
from cos435_citylearn.io import ensure_parent, write_csv_row, write_json
from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.run_id import build_run_id
from cos435_citylearn.runtime import build_environment_lock, utc_now_iso

class CityLearnGymWrapper(gym.Env):
    def __init__(self, env_bundle):
        super().__init__()
        self.citylearn_env = env_bundle.env

        # figure out obs / action sizes from the CityLearn spaces
        obs_dim = sum(box.shape[0] for box in self.citylearn_env.observation_space)
        act_dim = sum(box.shape[0] for box in self.citylearn_env.action_space)

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
        flat = np.concatenate([np.array(o, dtype=np.float32) for o in obs])
        return flat, {}

    def step(self, action):
        # split the flat action back into per-building chunks
        per_building_actions = []
        idx = 0
        for box in self.citylearn_env.action_space:
            dim = box.shape[0]
            a = np.clip(action[idx : idx + dim], box.low, box.high)
            per_building_actions.append(a.tolist())
            idx += dim

        result = self.citylearn_env.step(per_building_actions)

        if len(result) == 4:
            obs, rewards, terminated, info = result
            truncated = False
        else:
            obs, rewards, terminated, truncated, info = result

        flat_obs = np.concatenate([np.array(o, dtype=np.float32) for o in obs])
        total_reward = float(sum(rewards)) / max(len(rewards), 1)

        return flat_obs, total_reward, bool(terminated), bool(truncated), info or {}

class TrainingLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.curve_rows = []

    def _on_step(self):
        # log every 2048 steps (rollout) to keep csv manageable
        if self.num_timesteps % 2048 == 0:
            # grab mean reward from the SB3 logger if available
            mean_reward = None
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            self.curve_rows.append({
                "step": self.num_timesteps,
                "mean_reward": mean_reward,
            })
        return True

def run_ppo(
    config_path: str | Path = "configs/train/ppo/ppo_central_baseline.yaml",
    eval_config_path: str | Path = "configs/eval/default.yaml",
    output_root: str | Path | None = None,
    metrics_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    **kwargs,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_config = load_yaml(eval_config_path)

    # resolve reward function (reward_v0 = default)
    reward_version = config["reward"]["version"]
    reward_function = resolve_reward_function(reward_version)

    seed = int(config["training"]["seed"])
    total_timesteps = int(config["training"]["total_timesteps"])

    # build training env
    train_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=seed,
        central_agent=True,
        reward_function=reward_function,
    )
    train_env = CityLearnGymWrapper(train_bundle)

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

    print(f"Training PPO for {total_timesteps} timesteps...")
    log_callback = TrainingLogCallback()
    model.learn(total_timesteps=total_timesteps, callback=log_callback)

    variant = config["algorithm"].get("variant", "ppo_central_baseline")
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
    curve_path = run_dir / "training_curve.csv"

    # save model
    ensure_parent(model_path)
    model.save(str(model_path))

    # save training curve
    if log_callback.curve_rows:
        ensure_parent(curve_path)
        with curve_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "mean_reward"])
            writer.writeheader()
            writer.writerows(log_callback.curve_rows)

    # evaluate on new env
    print("Evaluating trained PPO agent...")
    eval_reward_function = resolve_reward_function(reward_version)
    eval_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=seed,
        central_agent=True,
        reward_function=eval_reward_function,
    )
    eval_env = CityLearnGymWrapper(eval_bundle)

    deterministic = bool(eval_config["evaluation"].get("deterministic", True))
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = eval_env.step(action)
        if truncated:
            done = True

    # compute metrics via pipeline
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

    # save
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(run_dir / "manifest.json", {
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "config_path": str(resolve_path(config_path)),
        "eval_config_path": str(resolve_path(eval_config_path)),
        "dataset_name": eval_bundle.dataset_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "model_path": str(model_path),
    })
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

    print(f"Average score: {metrics_payload['average_score']}")

    return {
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics.json"),
        "csv_path": str(metrics_dir / f"{run_id}.csv"),
        "run_id": run_id,
        "average_score": metrics_payload["average_score"],
        "model_path": str(model_path),
        "training_curve_path": str(curve_path),
    }

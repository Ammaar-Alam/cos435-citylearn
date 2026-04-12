from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from cos435_citylearn.config import load_yaml, resolve_path
from cos435_citylearn.env import CentralizedEnvAdapter, make_citylearn_env
from cos435_citylearn.eval import build_metrics_payload, flatten_metrics_row
from cos435_citylearn.io import write_csv_row, write_json
from cos435_citylearn.paths import RESULTS_DIR
from cos435_citylearn.run_id import build_run_id
from cos435_citylearn.runtime import build_environment_lock, utc_now_iso


def _import_from_path(path: str):
    module_name, symbol_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, symbol_name)


def run_rbc(
    config_path: str | Path = "configs/train/rbc/rbc_builtin.yaml",
    eval_config_path: str | Path = "configs/eval/default.yaml",
    output_root: str | Path | None = None,
    metrics_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_config = load_yaml(eval_config_path)
    env_bundle = make_citylearn_env(
        config["env"]["base_config"],
        f"configs/splits/{config['env']['split']}.yaml",
        seed=config["training"]["seed"],
        central_agent=True,
    )
    adapter = CentralizedEnvAdapter(env_bundle.env)
    controller_class = _import_from_path(config["algorithm"]["controller_class"])
    controller = controller_class(env_bundle.env)
    variant = config["algorithm"]["variant"]
    run_id = build_run_id(
        algo=config["algorithm"]["name"],
        variant=variant,
        split=config["env"]["split"],
        seed=env_bundle.seed,
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
    rollout_trace = []
    observations = adapter.reset()
    trace_limit = int(config["evaluation"].get("trace_limit", 96))
    step_index = 0

    while not adapter.done:
        actions = controller.predict(observations)
        result = adapter.step(actions)
        if config["evaluation"]["save_rollout_trace"] and step_index < trace_limit:
            rollout_trace.append(
                {
                    "step": step_index,
                    "actions": actions,
                    "rewards": result.rewards,
                    "terminated": result.terminated,
                }
            )
        observations = result.observations
        step_index += 1

    run_context = {
        "algorithm": config["algorithm"]["name"],
        "variant": variant,
        "split": config["env"]["split"],
        "seed": env_bundle.seed,
        "dataset_name": env_bundle.dataset_name,
        "run_id": run_id,
    }
    metrics_payload = build_metrics_payload(env_bundle.env, run_context)
    row = flatten_metrics_row(metrics_payload)
    manifest = {
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "config_path": str(resolve_path(config_path)),
        "eval_config_path": str(resolve_path(eval_config_path)),
        "schema_path": str(env_bundle.schema_path),
        "dataset_name": env_bundle.dataset_name,
        "seed": env_bundle.seed,
        "step_count": step_index,
    }

    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics.json", metrics_payload)
    if config["evaluation"]["save_rollout_trace"]:
        write_json(run_dir / "rollout_trace.json", rollout_trace)

    write_csv_row(metrics_dir / f"{run_id}.csv", row)
    write_json(
        manifests_dir / "environment_lock.json",
        build_environment_lock(
            {
                "dataset_name": env_bundle.dataset_name,
                "schema_path": str(env_bundle.schema_path),
                "seed": env_bundle.seed,
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
    }

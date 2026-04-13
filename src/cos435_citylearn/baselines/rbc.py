from __future__ import annotations

from collections import deque
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
from cos435_citylearn.ui_exports import (
    DashboardCapture,
    build_live_preview_payload,
    export_simulation_bundle,
)


def _import_from_path(path: str):
    module_name, symbol_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, symbol_name)


def _resolved_artifact_path(path_value: str, artifacts_root: str | Path | None) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    root = RESULTS_DIR if artifacts_root is None else Path(artifacts_root)
    return str((root / path).resolve())


def run_rbc(
    config_path: str | Path = "configs/train/rbc/rbc_builtin.yaml",
    eval_config_path: str | Path = "configs/eval/default.yaml",
    output_root: str | Path | None = None,
    metrics_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    ui_exports_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    artifact_id: str | None = None,
    job_id: str | None = None,
    job_dir: str | Path | None = None,
    progress_context: Any | None = None,
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
    decision_total = max(int(getattr(env_bundle.env, "time_steps", 0)) - 1, 0)
    export_enabled = bool(eval_config["evaluation"].get("export_simulation_data", True))
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
    trace_limit = int(config["evaluation"].get("trace_limit", 96))
    rollout_trace = []
    playback_trace: list[dict[str, Any]] = [] if export_enabled else []
    preview_trace: deque[dict[str, Any]] = deque(maxlen=trace_limit)
    observations = adapter.reset()
    capture = DashboardCapture(
        run_id=run_id,
        dataset_name=env_bundle.dataset_name,
        ui_exports_root=ui_exports_root,
        artifacts_root=artifacts_root,
        enabled=bool(eval_config["evaluation"].get("export_simulation_data", True)),
        capture_frames=bool(eval_config["evaluation"].get("capture_render_frames", True)),
        max_frames=int(eval_config["evaluation"].get("max_render_frames", 60)),
        frame_width=int(eval_config["evaluation"].get("render_frame_width", 960)),
    )
    capture.configure(env_bundle.env)
    step_index = 0

    run_context = {
        "algorithm": config["algorithm"]["name"],
        "variant": variant,
        "split": config["env"]["split"],
        "seed": env_bundle.seed,
        "dataset_name": env_bundle.dataset_name,
        "run_id": run_id,
    }
    preview_stride = max(1, int(config["evaluation"].get("preview_stride", 12)))

    if progress_context is not None:
        progress_context.start(
            phase="rollout",
            total=decision_total,
            label=f"{variant} on {config['env']['split']}",
        )

    while not adapter.done:
        actions = controller.predict(observations)
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
        if config["evaluation"]["save_rollout_trace"] and step_index < trace_limit:
            rollout_trace.append(frame_payload)
        observations = result.observations
        capture.maybe_capture(env=env_bundle.env, step_index=step_index, force=result.terminated)
        if progress_context is not None and (
            step_index == 0 or result.terminated or (step_index + 1) % preview_stride == 0
        ):
            preview_payload = build_live_preview_payload(
                env=env_bundle.env,
                run_id=run_id,
                run_context=run_context,
                rollout_trace=list(preview_trace),
                capture=capture,
                current_step=step_index,
                history_limit=trace_limit,
                ui_exports_root=ui_exports_root,
                artifacts_root=artifacts_root,
            )
            progress_context.update(
                phase="rollout",
                current=min(step_index + 1, decision_total),
                total=decision_total,
                label=f"{variant} rollout",
                preview_payload=preview_payload,
                run_id=run_id,
            )
        step_index += 1

    metrics_payload = build_metrics_payload(env_bundle.env, run_context)
    row = flatten_metrics_row(metrics_payload)
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
    if artifact_id:
        manifest["artifact_id"] = artifact_id
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

    payload = {
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics.json"),
        "csv_path": str(metrics_dir / f"{run_id}.csv"),
        "run_id": run_id,
        "average_score": metrics_payload["average_score"],
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

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import yaml

from cos435_citylearn.baselines import run_sac
from cos435_citylearn.config import load_yaml


class ConsoleProgress:
    def __init__(self, *, min_step_delta: int = 1000, min_seconds_delta: float = 20.0) -> None:
        self.min_step_delta = min_step_delta
        self.min_seconds_delta = min_seconds_delta
        self.started_at = time.time()
        self.last_step = -1
        self.last_print_at = 0.0

    def start(self, *, phase: str, total: int | None = None, label: str | None = None) -> None:
        print(f"[start] phase={phase} total={total} label={label}", flush=True)

    def update(
        self,
        *,
        phase: str,
        current: int,
        total: int | None,
        label: str | None = None,
        preview_payload: dict | None = None,
        run_id: str | None = None,
    ) -> None:
        now = time.time()
        should_print = (
            self.last_step < 0
            or current == total
            or current - self.last_step >= self.min_step_delta
            or now - self.last_print_at >= self.min_seconds_delta
        )
        if not should_print:
            return
        elapsed = now - self.started_at
        pct = (100.0 * current / total) if total else None
        pct_text = f"{pct:6.2f}%" if pct is not None else "   n/a"
        print(
            f"[progress] phase={phase} step={current}/{total} pct={pct_text} elapsed={elapsed:7.1f}s label={label}",
            flush=True,
        )
        self.last_step = current
        self.last_print_at = now

    def artifact(self, *, kind: str, path: str, label: str) -> None:
        print(f"[artifact] kind={kind} label={label} path={path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train/sac/sac_central_reward_v1.yaml",
        help="training config that defines the SAC controller shape",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/eval/official_released.yaml",
        help="evaluation config path",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="local checkpoint path to evaluate",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="split config stem under configs/splits without .yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed override used for run_id and environment construction",
    )
    parser.add_argument("--output-root", default=None, help="override results/runs root")
    parser.add_argument("--metrics-root", default=None, help="override results/metrics root")
    parser.add_argument("--manifests-root", default=None, help="override results/manifests root")
    parser.add_argument(
        "--ui-exports-root",
        default=None,
        help="override results/ui_exports root",
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="base path used to resolve relative artifact paths in payloads",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.split is not None:
        config["env"]["split"] = args.split
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    progress = ConsoleProgress()

    with tempfile.TemporaryDirectory(prefix="cos435_sac_eval_") as tmp_dir:
        tmp_config_path = Path(tmp_dir) / "config.yaml"
        tmp_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        try:
            payload = run_sac(
                config_path=tmp_config_path,
                eval_config_path=args.eval_config,
                checkpoint_path=args.checkpoint_path,
                output_root=args.output_root,
                metrics_root=args.metrics_root,
                manifests_root=args.manifests_root,
                ui_exports_root=args.ui_exports_root,
                artifacts_root=args.artifacts_root,
                progress_context=progress,
            )
        except ValueError as exc:
            split_name = str(config["env"]["split"])
            control_mode = str(config["algorithm"]["control_mode"])
            if (
                "observation schema is incompatible" in str(exc)
                and control_mode == "centralized"
                and split_name.startswith("phase_3_")
            ):
                raise ValueError(
                    "centralized SAC checkpoints trained on three-building public_dev are not portable to the released six-building phase_3 splits"
                ) from exc
            raise

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

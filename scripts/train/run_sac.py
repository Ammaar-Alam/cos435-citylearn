from __future__ import annotations

import argparse
import json

from cos435_citylearn.baselines import run_sac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train/sac/sac_central_baseline.yaml",
        help="training config path",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/eval/default.yaml",
        help="evaluation config path",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "override env.split (eg. phase_3_1). required when using "
            "--artifact-id to eval on a held-out split"
        ),
    )
    parser.add_argument(
        "--artifact-id",
        default=None,
        help="run_id of an existing SAC checkpoint to re-evaluate without training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override training.seed from the config (useful for multi-seed sweeps)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override training.learning_rate from the config",
    )
    parser.add_argument("--output-root", default=None, help="directory for run artifacts")
    parser.add_argument("--metrics-root", default=None, help="directory for metric CSV outputs")
    parser.add_argument("--manifests-root", default=None, help="directory for run manifests")
    parser.add_argument("--ui-exports-root", default=None, help="directory for UI export artifacts")
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="root that contains runs/, metrics/, manifests/ for artifact lookup",
    )
    parser.add_argument(
        "--imported-artifacts-root",
        default=None,
        help="root for imported artifacts when evaluating externally copied checkpoints",
    )
    parser.add_argument(
        "--allow-cross-reward-eval",
        action="store_true",
        help=(
            "opt into evaluating a checkpoint whose trained variant/reward/features "
            "version does not match the eval config (mismatch is recorded in the "
            "run manifest under runtime_label_mismatches)"
        ),
    )
    args = parser.parse_args()

    payload = run_sac(
        config_path=args.config,
        eval_config_path=args.eval_config,
        artifact_id=args.artifact_id,
        split_override=args.split,
        seed_override=args.seed,
        lr_override=args.lr,
        output_root=args.output_root,
        metrics_root=args.metrics_root,
        manifests_root=args.manifests_root,
        ui_exports_root=args.ui_exports_root,
        artifacts_root=args.artifacts_root,
        imported_artifacts_root=args.imported_artifacts_root,
        allow_cross_reward_eval=args.allow_cross_reward_eval,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from cos435_citylearn.config import load_yaml
from cos435_citylearn.api.schemas import (
    ArtifactDetail,
    ArtifactKind,
    ArtifactSummary,
    EvaluateArtifactRequest,
)
from cos435_citylearn.api.services.runner_registry import get_runner
from cos435_citylearn.api.settings import ApiSettings
from cos435_citylearn.io import ensure_parent, write_json_atomic
from cos435_citylearn.runtime import utc_now_iso


def _load_sac_checkpoint_tools():
    from cos435_citylearn.algorithms.sac import (
        resolve_imported_checkpoint_path,
        safe_load_checkpoint_payload,
        validate_checkpoint_env_compatibility,
        validate_checkpoint_runner_compatibility,
    )

    return (
        resolve_imported_checkpoint_path,
        safe_load_checkpoint_payload,
        validate_checkpoint_env_compatibility,
        validate_checkpoint_runner_compatibility,
    )


def _load_ppo_checkpoint_tools():
    # ``resolve_imported_checkpoint_path`` lives in the SAC package today but
    # is algorithm-agnostic (it just reads ``artifact.json`` and resolves the
    # ``file_path`` entry). PPO reuses it here rather than duplicating the
    # layout logic; if it ever sprouts SAC-specific behavior, lift it to a
    # shared module before that happens.
    from cos435_citylearn.algorithms.ppo import (
        safe_load_ppo_checkpoint_payload,
        validate_ppo_checkpoint_env_compatibility,
        validate_ppo_checkpoint_runner_compatibility,
    )
    from cos435_citylearn.algorithms.sac.checkpoints import (
        resolve_imported_checkpoint_path,
    )

    return (
        resolve_imported_checkpoint_path,
        safe_load_ppo_checkpoint_payload,
        validate_ppo_checkpoint_env_compatibility,
        validate_ppo_checkpoint_runner_compatibility,
    )


def _load_central_ppo_sidecar_tools():
    from cos435_citylearn.baselines.ppo import (
        _load_central_ppo_sidecar,
        _validate_central_ppo_sidecar,
    )

    return _load_central_ppo_sidecar, _validate_central_ppo_sidecar


def _load_citylearn_env_factory():
    from cos435_citylearn.env import make_citylearn_env

    return make_citylearn_env


@dataclass(frozen=True)
class ImportedArtifactRecord:
    artifact_id: str
    artifact_kind: ArtifactKind
    label: str
    source_filename: str
    imported_at: str
    algorithm: str
    runner_id: str | None
    status: str
    file_path: str
    notes: str | None
    evaluable: bool
    playback_path: str | None = None
    simulation_dir: str | None = None


class ArtifactStore:
    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self.settings.imported_artifacts_root.mkdir(parents=True, exist_ok=True)

    def _artifact_dir(self, artifact_id: str) -> Path:
        return self.settings.imported_artifacts_root / artifact_id

    def _artifact_path(self, artifact_id: str) -> Path:
        return self._artifact_dir(artifact_id) / "artifact.json"

    def _normalize_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        for root in (self.settings.artifacts_root, self.settings.repo_root):
            try:
                return str(path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                continue
        return str(path)

    def _read_record(self, artifact_id: str) -> ImportedArtifactRecord:
        path = self._artifact_path(artifact_id)
        if not path.exists():
            raise KeyError(f"unknown artifact: {artifact_id}")
        return ImportedArtifactRecord(**json.loads(path.read_text()))

    def _detect_playback_path(self, stored_path: Path) -> str | None:
        if stored_path.suffix.lower() != ".json":
            return None

        try:
            payload = json.loads(stored_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None
        if {"run_id", "trace", "buildings"} <= set(payload):
            return self._normalize_path(stored_path)
        return None

    async def _stream_to_disk(self, file: UploadFile, dest: Path) -> None:
        """Stream an UploadFile to ``dest`` in 1 MiB chunks and close it."""
        with dest.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        await file.close()

    async def import_upload(
        self,
        *,
        file: UploadFile,
        artifact_kind: ArtifactKind,
        label: str,
        notes: str | None,
        runner_id: str | None,
        algorithm: str | None,
        extra_files: list[UploadFile] | None = None,
    ) -> ArtifactDetail:
        artifact_id = f"artifact_{uuid4().hex[:10]}"
        artifact_dir = self._artifact_dir(artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(file.filename or "artifact.bin").name
        stored_path = ensure_parent(artifact_dir / filename)
        await self._stream_to_disk(file, stored_path)

        # Companion files (e.g. ``vec_normalize.pkl`` for centralized PPO)
        # land in the same artifact dir as siblings. We reject duplicate
        # filenames up front because the second write would silently clobber
        # the first, and users would ship an artifact that looks complete
        # but is missing a required file.
        if extra_files:
            seen_names = {filename}
            for extra in extra_files:
                extra_name = Path(extra.filename or "").name
                if not extra_name:
                    raise ValueError("extra_files entry is missing a filename")
                if extra_name in seen_names:
                    raise ValueError(
                        f"duplicate filename in upload: {extra_name!r}. "
                        "Each file in a single import must have a unique name."
                    )
                seen_names.add(extra_name)
                extra_path = ensure_parent(artifact_dir / extra_name)
                await self._stream_to_disk(extra, extra_path)

        evaluable = False
        status = "validated"
        effective_algorithm = algorithm or "unknown"
        if runner_id:
            spec = get_runner(runner_id)
            effective_algorithm = spec.algorithm
            evaluable = spec.supports_checkpoint_eval and artifact_kind == "checkpoint"
            status = "evaluable" if evaluable else "registered"

        playback_path = self._detect_playback_path(stored_path)

        if playback_path and status == "validated":
            status = "inspectable"

        record = ImportedArtifactRecord(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            label=label.strip() or filename,
            source_filename=filename,
            imported_at=utc_now_iso(),
            algorithm=effective_algorithm,
            runner_id=runner_id,
            status=status,
            file_path=self._normalize_path(stored_path) or str(stored_path),
            notes=notes,
            evaluable=evaluable,
            playback_path=playback_path,
        )
        write_json_atomic(self._artifact_path(artifact_id), asdict(record))
        return self.get_artifact(artifact_id)

    def list_artifacts(self) -> list[ArtifactSummary]:
        artifacts = []
        for path in self.settings.imported_artifacts_root.glob("*/artifact.json"):
            record = ImportedArtifactRecord(**json.loads(path.read_text()))
            artifacts.append(
                ArtifactSummary(
                    artifact_id=record.artifact_id,
                    artifact_kind=record.artifact_kind,
                    label=record.label,
                    source_filename=record.source_filename,
                    imported_at=record.imported_at,
                    algorithm=record.algorithm,
                    runner_id=record.runner_id,
                    status=record.status,
                    evaluable=record.evaluable,
                    playback_path=record.playback_path,
                    simulation_dir=record.simulation_dir,
                )
            )
        artifacts.sort(key=lambda item: item.imported_at, reverse=True)
        return artifacts

    def get_artifact(self, artifact_id: str) -> ArtifactDetail:
        record = self._read_record(artifact_id)
        return ArtifactDetail(
            artifact_id=record.artifact_id,
            artifact_kind=record.artifact_kind,
            label=record.label,
            source_filename=record.source_filename,
            imported_at=record.imported_at,
            algorithm=record.algorithm,
            runner_id=record.runner_id,
            status=record.status,
            evaluable=record.evaluable,
            file_path=record.file_path,
            notes=record.notes,
            playback_path=record.playback_path,
            simulation_dir=record.simulation_dir,
        )

    def build_evaluation_request(
        self, artifact_id: str, request: EvaluateArtifactRequest
    ) -> dict[str, Any]:
        record = self._read_record(artifact_id)
        if not record.evaluable or not record.runner_id:
            raise ValueError("artifact is not evaluable yet")

        spec = get_runner(record.runner_id)
        if spec.algorithm == "sac" and record.artifact_kind == "checkpoint":
            (
                resolve_imported_checkpoint_path,
                safe_load_checkpoint_payload,
                validate_checkpoint_env_compatibility,
                validate_checkpoint_runner_compatibility,
            ) = _load_sac_checkpoint_tools()
            make_citylearn_env = _load_citylearn_env_factory()
            config = load_yaml(spec.config_path)
            if request.seed is not None:
                config["training"]["seed"] = int(request.seed)
            if request.split is not None:
                config["env"]["split"] = request.split

            checkpoint_path = resolve_imported_checkpoint_path(
                artifact_id=artifact_id,
                imported_artifacts_root=self.settings.imported_artifacts_root,
                artifacts_root=self.settings.artifacts_root,
            )
            checkpoint_payload = safe_load_checkpoint_payload(checkpoint_path)
            validate_checkpoint_runner_compatibility(
                checkpoint_payload,
                config,
                allow_cross_reward_eval=request.allow_cross_reward_eval,
            )
            env_bundle = make_citylearn_env(
                config["env"]["base_config"],
                f"configs/splits/{config['env']['split']}.yaml",
                seed=config["training"]["seed"],
                central_agent=config["algorithm"]["control_mode"] == "centralized",
            )
            validate_checkpoint_env_compatibility(
                checkpoint_payload,
                observation_names=env_bundle.env.observation_names,
                action_names=env_bundle.env.action_names,
            )
        elif spec.algorithm == "ppo" and record.artifact_kind == "checkpoint":
            # Mirror the SAC preflight: fail fast at the API layer with 400
            # instead of letting the worker raise halfway through launch.
            # Central PPO ships SB3 zip + sidecar JSON (no torch payload to
            # load); shared PPO ships a torch .pt with embedded metadata.
            config = load_yaml(spec.config_path)
            if request.seed is not None:
                config["training"]["seed"] = int(request.seed)
            if request.split is not None:
                config["env"]["split"] = request.split

            (
                resolve_imported_checkpoint_path,
                safe_load_ppo_checkpoint_payload,
                validate_ppo_checkpoint_env_compatibility,
                validate_ppo_checkpoint_runner_compatibility,
            ) = _load_ppo_checkpoint_tools()
            checkpoint_path = resolve_imported_checkpoint_path(
                artifact_id=artifact_id,
                imported_artifacts_root=self.settings.imported_artifacts_root,
                artifacts_root=self.settings.artifacts_root,
            )

            control_mode = config["algorithm"]["control_mode"]
            if control_mode == "centralized":
                vec_normalize_path = checkpoint_path.parent / "vec_normalize.pkl"
                if not vec_normalize_path.exists():
                    raise FileNotFoundError(
                        "VecNormalize stats (vec_normalize.pkl) not found alongside imported "
                        f"central PPO model at {checkpoint_path.parent}. Re-import via the "
                        "dashboard and attach vec_normalize.pkl as a companion file on the "
                        "upload form (the 'extra_files' field); centralized PPO cannot "
                        "evaluate without the observation normalization stats."
                    )
                # SB3 zip: validate the sidecar instead of loading the zip. The
                # worker still checks again on load, but preflighting here
                # surfaces a misrouted artifact (reward_v1 under reward_v2
                # runner, wrong variant label, etc.) before the job queues.
                (
                    _load_central_ppo_sidecar,
                    _validate_central_ppo_sidecar,
                ) = _load_central_ppo_sidecar_tools()
                sidecar = _load_central_ppo_sidecar(checkpoint_path)
                _validate_central_ppo_sidecar(
                    sidecar,
                    config,
                    allow_cross_reward_eval=request.allow_cross_reward_eval,
                    artifact_id=artifact_id,
                )
            else:
                # Shared PPO: same shape as SAC -- torch payload + env preflight.
                make_citylearn_env = _load_citylearn_env_factory()
                checkpoint_payload = safe_load_ppo_checkpoint_payload(checkpoint_path)
                validate_ppo_checkpoint_runner_compatibility(
                    checkpoint_payload,
                    config,
                    allow_cross_reward_eval=request.allow_cross_reward_eval,
                )
                env_bundle = make_citylearn_env(
                    config["env"]["base_config"],
                    f"configs/splits/{config['env']['split']}.yaml",
                    seed=config["training"]["seed"],
                    central_agent=False,
                )
                validate_ppo_checkpoint_env_compatibility(
                    checkpoint_payload,
                    observation_names=env_bundle.env.observation_names,
                    action_names=env_bundle.env.action_names,
                )

        return {
            "runner_id": record.runner_id,
            "artifact_id": artifact_id,
            "seed": request.seed,
            "split": request.split,
            "trace_limit": request.trace_limit,
            "capture_render_frames": request.capture_render_frames,
            "max_render_frames": request.max_render_frames,
            "render_frame_width": request.render_frame_width,
            "allow_cross_reward_eval": request.allow_cross_reward_eval,
        }

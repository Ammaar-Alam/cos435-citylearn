from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from cos435_citylearn.io import ensure_parent, write_json
from cos435_citylearn.paths import REPO_ROOT, RESULTS_DIR
from cos435_citylearn.runtime import utc_now_iso

UI_EXPORTS_ROOT = RESULTS_DIR / "ui_exports"
SIMULATION_ROOT = RESULTS_DIR / "ui_exports" / "SimulationData"
MEDIA_ROOT = RESULTS_DIR / "ui_exports" / "media"


def _to_float_list(values: Any, *, scale: float = 1.0) -> list[float]:
    if values is None:
        return []

    if hasattr(values, "tolist"):
        values = values.tolist()

    return [float(value) * scale for value in values]


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []

    if hasattr(values, "tolist"):
        values = values.tolist()

    return [int(value) for value in values]


def _trim(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return values
    return values[:limit]


def _relative_artifact_path(path: Path | None, *, relative_root: Path | None = None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    if relative_root is not None:
        try:
            return str(resolved.relative_to(relative_root.resolve()).as_posix())
        except ValueError:
            pass
    try:
        return str(resolved.relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(resolved)


def _resolve_ui_exports_root(ui_exports_root: str | Path | None) -> Path:
    return UI_EXPORTS_ROOT if ui_exports_root is None else Path(ui_exports_root)


def _hourly_timestamps(steps: int, start: datetime | None = None) -> list[str]:
    anchor = datetime(2023, 1, 1, tzinfo=timezone.utc) if start is None else start
    return [
        (anchor + timedelta(hours=index)).isoformat().replace("+00:00", "Z")
        for index in range(steps)
    ]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _build_kpi_rows(env: Any) -> list[dict[str, Any]]:
    frame = env.evaluate()
    names = list(frame["name"].unique())
    rows = []

    for metric_name in frame["cost_function"].unique():
        metric_rows = frame[frame["cost_function"] == metric_name]
        row = {"KPI": metric_name}

        for name in names:
            match = metric_rows[metric_rows["name"] == name]
            value = None if match.empty else float(match.iloc[0]["value"])
            row[name] = value

        rows.append(row)

    return rows


def _pricing_rows(building: Any, timestamps: list[str]) -> list[dict[str, Any]]:
    actual = _to_float_list(building.pricing.electricity_pricing)
    predicted_6h = _to_float_list(building.pricing.electricity_pricing_predicted_6h)
    predicted_12h = _to_float_list(building.pricing.electricity_pricing_predicted_12h)
    predicted_24h = _to_float_list(building.pricing.electricity_pricing_predicted_24h)
    carbon = _to_float_list(building.carbon_intensity.carbon_intensity)

    return [
        {
            "timestamp": timestamps[index],
            "electricity_pricing-$/kWh": actual[index],
            "electricity_pricing_predicted_1-$/kWh": predicted_6h[index],
            "electricity_pricing_predicted_2-$/kWh": predicted_12h[index],
            "electricity_pricing_predicted_3-$/kWh": predicted_24h[index],
            "carbon_intensity-kgCO2/kWh": carbon[index],
        }
        for index in range(len(timestamps))
    ]


def _building_rows(building: Any, timestamps: list[str]) -> list[dict[str, Any]]:
    solar = _to_float_list(building.solar_generation, scale=-1.0)
    non_shiftable = _to_float_list(building.non_shiftable_load)
    net_load = _to_float_list(building.net_electricity_consumption)
    cooling = _to_float_list(building.cooling_demand)
    dhw = _to_float_list(building.dhw_demand)
    indoor = _to_float_list(building.indoor_dry_bulb_temperature)
    set_point = _to_float_list(building.indoor_dry_bulb_temperature_set_point)
    occupants = _to_float_list(building.occupant_count)
    outages = _to_int_list(building.power_outage_signal)

    return [
        {
            "timestamp": timestamps[index],
            "Energy Production from PV-kWh": solar[index],
            "Energy Production from EV-kWh": 0.0,
            "Non-shiftable Load-kWh": non_shiftable[index],
            "Net Electricity Consumption-kWh": net_load[index],
            "Cooling Demand-kWh": cooling[index],
            "DHW Demand-kWh": dhw[index],
            "Indoor Dry Bulb Temperature-C": indoor[index],
            "Indoor Set Point-C": set_point[index],
            "Occupant Count": occupants[index],
            "Power Outage": outages[index],
        }
        for index in range(len(timestamps))
    ]


def _battery_rows(building: Any, timestamps: list[str]) -> list[dict[str, Any]]:
    soc = _to_float_list(building.electrical_storage.soc, scale=100.0)
    delta = _to_float_list(building.electrical_storage.energy_balance)

    return [
        {
            "timestamp": timestamps[index],
            "Battery Soc-%": soc[index],
            "Battery (Dis)Charge-kWh": delta[index],
        }
        for index in range(len(timestamps))
    ]


def _community_rows(env: Any, timestamps: list[str]) -> list[dict[str, Any]]:
    net_load = _to_float_list(getattr(env, "net_electricity_consumption", []))
    costs = _to_float_list(getattr(env, "net_electricity_consumption_cost", []))
    emissions = _to_float_list(getattr(env, "net_electricity_consumption_emission", []))

    return [
        {
            "timestamp": timestamps[index],
            "Net Electricity Consumption-kWh": net_load[index] if index < len(net_load) else 0.0,
            "Net Electricity Consumption Cost-$": costs[index] if index < len(costs) else 0.0,
            "Net Electricity Consumption Emissions-kgCO2": emissions[index]
            if index < len(emissions)
            else 0.0,
        }
        for index in range(len(timestamps))
    ]


def _playback_payload(
    *,
    env: Any,
    run_context: dict[str, Any],
    metrics_payload: dict[str, Any],
    rollout_trace: list[dict[str, Any]],
    timestamps: list[str],
    simulation_dir: Path,
    media_manifest: dict[str, Any],
    artifacts_root: Path | None = None,
    series_limit: int | None = None,
) -> dict[str, Any]:
    buildings = []

    for building in env.buildings:
        buildings.append(
            {
                "name": building.name,
                "series": {
                    "net_electricity_consumption": _trim(
                        _to_float_list(building.net_electricity_consumption), series_limit
                    ),
                    "non_shiftable_load": _trim(
                        _to_float_list(building.non_shiftable_load), series_limit
                    ),
                    "solar_generation": _trim(
                        _to_float_list(building.solar_generation, scale=-1.0), series_limit
                    ),
                    "cooling_demand": _trim(_to_float_list(building.cooling_demand), series_limit),
                    "dhw_demand": _trim(_to_float_list(building.dhw_demand), series_limit),
                    "indoor_temperature": _trim(
                        _to_float_list(building.indoor_dry_bulb_temperature), series_limit
                    ),
                    "temperature_set_point": _trim(
                        _to_float_list(
                        building.indoor_dry_bulb_temperature_set_point
                        ),
                        series_limit,
                    ),
                    "occupant_count": _trim(_to_float_list(building.occupant_count), series_limit),
                    "power_outage": _trim(_to_int_list(building.power_outage_signal), series_limit),
                    "battery_soc": _trim(
                        _to_float_list(building.electrical_storage.soc, scale=100.0), series_limit
                    ),
                    "battery_delta": _trim(
                        _to_float_list(building.electrical_storage.energy_balance), series_limit
                    ),
                    "electricity_pricing": _trim(
                        _to_float_list(building.pricing.electricity_pricing), series_limit
                    ),
                    "electricity_pricing_predicted_6h": _trim(
                        _to_float_list(building.pricing.electricity_pricing_predicted_6h),
                        series_limit,
                    ),
                    "electricity_pricing_predicted_12h": _trim(
                        _to_float_list(building.pricing.electricity_pricing_predicted_12h),
                        series_limit,
                    ),
                    "electricity_pricing_predicted_24h": _trim(
                        _to_float_list(building.pricing.electricity_pricing_predicted_24h),
                        series_limit,
                    ),
                    "carbon_intensity": _trim(
                        _to_float_list(building.carbon_intensity.carbon_intensity), series_limit
                    ),
                },
            }
        )

    effective_timestamps = timestamps if series_limit is None else timestamps[:series_limit]
    decision_steps = (
        len(rollout_trace)
        if series_limit is None
        else min(len(rollout_trace), series_limit)
    )
    return {
        **run_context,
        "generated_at": utc_now_iso(),
        "timestamps": effective_timestamps,
        "time_steps": len(effective_timestamps),
        "episode_total_steps": len(timestamps),
        "decision_steps": decision_steps,
        "building_names": [building["name"] for building in buildings],
        "action_names": getattr(env, "action_names", []),
        "metrics": metrics_payload,
        "district": {
            "net_electricity_consumption": _trim(
                _to_float_list(env.net_electricity_consumption), series_limit
            ),
            "net_electricity_consumption_cost": _trim(
                _to_float_list(getattr(env, "net_electricity_consumption_cost", [])), series_limit
            ),
            "net_electricity_consumption_emission": _trim(
                _to_float_list(getattr(env, "net_electricity_consumption_emission", [])),
                series_limit,
            ),
        },
        "buildings": buildings,
        "trace": rollout_trace if series_limit is None else rollout_trace[:series_limit],
        "ui_export": {
            "simulation_dir": _relative_artifact_path(simulation_dir, relative_root=artifacts_root),
        },
        "media": media_manifest,
    }


@dataclass
class DashboardCapture:
    run_id: str
    dataset_name: str
    ui_exports_root: str | Path | None = None
    artifacts_root: str | Path | None = None
    enabled: bool = True
    capture_frames: bool = True
    max_frames: int = 60
    frame_width: int = 960

    def __post_init__(self) -> None:
        self.ui_exports_root = _resolve_ui_exports_root(self.ui_exports_root)
        self.artifacts_root = (
            self.ui_exports_root.parent if self.artifacts_root is None else Path(self.artifacts_root)
        )
        self.simulation_dir = self.ui_exports_root / "SimulationData" / self.run_id
        self.media_dir = self.ui_exports_root / "media" / self.run_id
        self.frame_dir = self.media_dir / "frames"
        self.frame_count = 0
        self.frame_stride: int | None = None
        self.frame_paths: list[Path] = []
        self.poster_path: Path | None = None
        self.gif_path: Path | None = None

    def configure(self, env: Any) -> None:
        if not self.capture_frames:
            return

        total_steps = max(int(getattr(env, "time_steps", 0)), 1)
        self.frame_stride = max(1, total_steps // max(self.max_frames, 1))

    def maybe_capture(self, *, env: Any, step_index: int, force: bool = False) -> None:
        if not self.enabled or not self.capture_frames:
            return

        if self.frame_stride is None:
            self.configure(env)

        if not force and step_index % self.frame_stride != 0:
            return

        frame = env.render()
        image = Image.fromarray(frame)
        image = image.convert("RGB")

        if self.frame_width and image.width > self.frame_width:
            scale = self.frame_width / image.width
            resized = (self.frame_width, max(1, int(image.height * scale)))
            image = image.resize(resized, Image.Resampling.LANCZOS)

        target = ensure_parent(self.frame_dir / f"frame_{self.frame_count:04d}.jpg")
        image.save(target, quality=85, optimize=True)
        self.frame_paths.append(target)
        self.frame_count += 1

        if self.poster_path is None:
            self.poster_path = target

    def finalize_media(self) -> dict[str, Any]:
        if not self.frame_paths:
            return {
                "frame_count": 0,
                "frame_stride": self.frame_stride,
                "poster_path": None,
                "gif_path": None,
                "frames": [],
            }

        images = [Image.open(path) for path in self.frame_paths]
        gif_path = self.media_dir / "playback.gif"
        ensure_parent(gif_path)
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=140,
            loop=0,
            optimize=True,
        )
        self.gif_path = gif_path

        return {
            "frame_count": self.frame_count,
            "frame_stride": self.frame_stride,
            "poster_path": _relative_artifact_path(self.poster_path, relative_root=self.artifacts_root),
            "gif_path": _relative_artifact_path(self.gif_path, relative_root=self.artifacts_root),
            "frames": [
                _relative_artifact_path(path, relative_root=self.artifacts_root)
                for path in self.frame_paths
            ],
        }

    def snapshot_media(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "frame_stride": self.frame_stride,
            "poster_path": _relative_artifact_path(self.poster_path, relative_root=self.artifacts_root),
            "gif_path": _relative_artifact_path(self.gif_path, relative_root=self.artifacts_root),
            "frames": [
                _relative_artifact_path(path, relative_root=self.artifacts_root)
                for path in self.frame_paths
            ],
        }


def build_live_preview_payload(
    *,
    env: Any,
    run_id: str,
    run_context: dict[str, Any],
    rollout_trace: list[dict[str, Any]],
    capture: DashboardCapture,
    current_step: int,
    ui_exports_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
) -> dict[str, Any]:
    series_limit = current_step + 1
    total_steps = int(getattr(env, "time_steps", series_limit))
    timestamps = _hourly_timestamps(total_steps)
    resolved_ui_exports_root = _resolve_ui_exports_root(ui_exports_root)
    resolved_artifacts_root = (
        resolved_ui_exports_root.parent if artifacts_root is None else Path(artifacts_root)
    )
    simulation_dir = resolved_ui_exports_root / "SimulationData" / run_id
    payload = _playback_payload(
        env=env,
        run_context=run_context,
        metrics_payload={},
        rollout_trace=rollout_trace,
        timestamps=timestamps,
        simulation_dir=simulation_dir,
        media_manifest=capture.snapshot_media(),
        artifacts_root=resolved_artifacts_root,
        series_limit=series_limit,
    )
    payload["ui_export"]["simulation_dir"] = _relative_artifact_path(
        simulation_dir,
        relative_root=resolved_artifacts_root,
    )
    payload["preview_step"] = current_step
    return payload


def export_simulation_bundle(
    *,
    env: Any,
    run_id: str,
    run_context: dict[str, Any],
    metrics_payload: dict[str, Any],
    rollout_trace: list[dict[str, Any]],
    capture: DashboardCapture | None = None,
    ui_exports_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_ui_exports_root = _resolve_ui_exports_root(ui_exports_root)
    resolved_artifacts_root = (
        resolved_ui_exports_root.parent if artifacts_root is None else Path(artifacts_root)
    )
    total_steps = int(
        getattr(env, "time_steps", len(env.buildings[0].net_electricity_consumption))
    )
    timestamps = _hourly_timestamps(total_steps)
    simulation_dir = resolved_ui_exports_root / "SimulationData" / run_id
    simulation_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    community_path = simulation_dir / "exported_data_community_ep0.csv"
    _write_csv(
        community_path,
        [
            "timestamp",
            "Net Electricity Consumption-kWh",
            "Net Electricity Consumption Cost-$",
            "Net Electricity Consumption Emissions-kgCO2",
        ],
        _community_rows(env, timestamps),
    )
    exported_files.append(community_path)

    for index, building in enumerate(env.buildings, start=1):
        building_path = simulation_dir / f"exported_data_building_{index}_ep0.csv"
        battery_path = simulation_dir / f"exported_data_building_{index}_battery_ep0.csv"

        _write_csv(
            building_path,
            [
                "timestamp",
                "Energy Production from PV-kWh",
                "Energy Production from EV-kWh",
                "Non-shiftable Load-kWh",
                "Net Electricity Consumption-kWh",
                "Cooling Demand-kWh",
                "DHW Demand-kWh",
                "Indoor Dry Bulb Temperature-C",
                "Indoor Set Point-C",
                "Occupant Count",
                "Power Outage",
            ],
            _building_rows(building, timestamps),
        )
        _write_csv(
            battery_path,
            ["timestamp", "Battery Soc-%", "Battery (Dis)Charge-kWh"],
            _battery_rows(building, timestamps),
        )
        exported_files.extend([building_path, battery_path])

    pricing_path = simulation_dir / "exported_data_pricing_ep0.csv"
    _write_csv(
        pricing_path,
        [
            "timestamp",
            "electricity_pricing-$/kWh",
            "electricity_pricing_predicted_1-$/kWh",
            "electricity_pricing_predicted_2-$/kWh",
            "electricity_pricing_predicted_3-$/kWh",
            "carbon_intensity-kgCO2/kWh",
        ],
        _pricing_rows(env.buildings[0], timestamps),
    )
    exported_files.append(pricing_path)

    kpi_path = simulation_dir / "exported_kpis.csv"
    kpi_rows = _build_kpi_rows(env)
    _write_csv(
        kpi_path,
        ["KPI", *list(env.evaluate()["name"].unique())],
        kpi_rows,
    )
    exported_files.append(kpi_path)

    media_manifest = capture.finalize_media() if capture is not None else {
        "frame_count": 0,
        "frame_stride": None,
        "poster_path": None,
        "gif_path": None,
        "frames": [],
    }
    playback_payload = _playback_payload(
        env=env,
        run_context=run_context,
        metrics_payload=metrics_payload,
        rollout_trace=rollout_trace,
        timestamps=timestamps,
        simulation_dir=simulation_dir,
        media_manifest=media_manifest,
        artifacts_root=resolved_artifacts_root,
    )
    playback_path = write_json(
        resolved_ui_exports_root / "playback" / f"{run_id}.json",
        playback_payload,
    )
    export_manifest_path = write_json(
        resolved_ui_exports_root / "manifests" / f"{run_id}.json",
        {
            "generated_at": utc_now_iso(),
            "run_id": run_id,
            "dataset_name": run_context["dataset_name"],
            "simulation_dir": _relative_artifact_path(simulation_dir, relative_root=resolved_artifacts_root),
            "playback_path": _relative_artifact_path(playback_path, relative_root=resolved_artifacts_root),
            "media": media_manifest,
            "files": [
                _relative_artifact_path(path, relative_root=resolved_artifacts_root)
                for path in exported_files
            ],
        },
    )

    return {
        "simulation_dir": _relative_artifact_path(simulation_dir, relative_root=resolved_artifacts_root)
        or str(simulation_dir),
        "playback_path": _relative_artifact_path(playback_path, relative_root=resolved_artifacts_root)
        or str(playback_path),
        "export_manifest_path": _relative_artifact_path(export_manifest_path, relative_root=resolved_artifacts_root)
        or str(export_manifest_path),
        "media": media_manifest,
    }

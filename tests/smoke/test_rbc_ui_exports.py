import csv
import re
from pathlib import Path

import pytest
import yaml

from cos435_citylearn.baselines import run_rbc
from cos435_citylearn.config import resolve_path


@pytest.mark.smoke
def test_rbc_writes_simulation_bundle(tmp_path: Path) -> None:
    eval_config = yaml.safe_load(Path("configs/eval/default.yaml").read_text())
    eval_config["evaluation"]["capture_render_frames"] = False
    eval_config["evaluation"]["max_render_frames"] = 8
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))

    payload = run_rbc(eval_config_path=eval_path)
    simulation_dir = resolve_path(payload["simulation_dir"])
    playback_path = resolve_path(payload["playback_path"])
    expected_files = {
        "exported_kpis.csv",
        "exported_data_community_ep0.csv",
        "exported_data_pricing_ep0.csv",
        "exported_data_building_1_ep0.csv",
        "exported_data_building_1_battery_ep0.csv",
        "exported_data_building_2_ep0.csv",
        "exported_data_building_2_battery_ep0.csv",
        "exported_data_building_3_ep0.csv",
        "exported_data_building_3_battery_ep0.csv",
    }

    assert expected_files.issubset({path.name for path in simulation_dir.iterdir()})

    with (simulation_dir / "exported_kpis.csv").open(newline="") as handle:
        kpi_rows = list(csv.DictReader(handle))

    assert {"KPI", "District", "Building_1", "Building_2", "Building_3"} <= set(
        kpi_rows[0].keys()
    )

    with (simulation_dir / "exported_data_pricing_ep0.csv").open(newline="") as handle:
        pricing_rows = list(csv.DictReader(handle))

    assert {
        "timestamp",
        "electricity_pricing-$/kWh",
        "electricity_pricing_predicted_1-$/kWh",
        "electricity_pricing_predicted_2-$/kWh",
        "electricity_pricing_predicted_3-$/kWh",
        "carbon_intensity-kgCO2/kWh",
    } <= set(pricing_rows[0].keys())

    with (simulation_dir / "exported_data_building_1_ep0.csv").open(newline="") as handle:
        building_rows = list(csv.DictReader(handle))

    assert {
        "timestamp",
        "Energy Production from PV-kWh",
        "Net Electricity Consumption-kWh",
        "Cooling Demand-kWh",
        "DHW Demand-kWh",
        "Indoor Dry Bulb Temperature-C",
        "Occupant Count",
        "Power Outage",
    } <= set(building_rows[0].keys())

    with (simulation_dir / "exported_data_building_1_battery_ep0.csv").open(newline="") as handle:
        battery_rows = list(csv.DictReader(handle))

    assert {"timestamp", "Battery Soc-%", "Battery (Dis)Charge-kWh"} <= set(
        battery_rows[0].keys()
    )
    assert playback_path.exists()


@pytest.mark.smoke
def test_rbc_simulation_bundle_matches_official_ui_upload_contract(tmp_path: Path) -> None:
    eval_config = yaml.safe_load(Path("configs/eval/default.yaml").read_text())
    eval_config["evaluation"]["capture_render_frames"] = False
    eval_config["evaluation"]["max_render_frames"] = 8
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False))

    payload = run_rbc(eval_config_path=eval_path)
    simulation_dir = resolve_path(payload["simulation_dir"])

    root_name = simulation_dir.parent.name
    assert root_name == "SimulationData"

    grouped_files: dict[str, dict[str, Path]] = {}
    for file_path in simulation_dir.iterdir():
        webkit_relative_path = f"{root_name}/{simulation_dir.name}/{file_path.name}"
        path_parts = webkit_relative_path.split("/")
        assert len(path_parts) == 3
        folder_name = path_parts[1]
        file_name = path_parts[2]
        grouped_files.setdefault(folder_name, {})[file_name] = file_path

    assert simulation_dir.name in grouped_files
    file_map = grouped_files[simulation_dir.name]
    assert "exported_kpis.csv" in file_map

    with file_map["exported_kpis.csv"].open(newline="") as handle:
        kpi_rows = list(csv.DictReader(handle))

    assert kpi_rows
    assert {"KPI", "District", "Building_1", "Building_2", "Building_3"} <= set(
        kpi_rows[0].keys()
    )

    parsed_data_files: dict[str, list[dict[str, str]]] = {}
    available_episodes: set[str] = set()
    for file_name, file_path in file_map.items():
        if re.search(r"kpi(s)?", file_name, re.IGNORECASE):
            continue

        cleaned_name = file_name.replace("exported_data_", "").rsplit(".", 1)[0]
        match = re.search(r"_?(ep\d+)", cleaned_name, re.IGNORECASE)
        if match:
            available_episodes.add(match.group(1))

        with file_path.open(newline="") as handle:
            parsed_data_files[cleaned_name] = list(csv.DictReader(handle))

    assert available_episodes == {"ep0"}
    assert {
        "building_1_ep0",
        "building_1_battery_ep0",
        "building_2_ep0",
        "building_2_battery_ep0",
        "building_3_ep0",
        "building_3_battery_ep0",
        "pricing_ep0",
    } <= set(parsed_data_files)
    assert parsed_data_files["building_1_ep0"]
    assert parsed_data_files["building_1_battery_ep0"]
    assert parsed_data_files["pricing_ep0"]

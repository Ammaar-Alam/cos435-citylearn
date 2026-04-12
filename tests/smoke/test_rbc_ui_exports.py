import csv
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

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from cos435_citylearn.paths import RESULTS_DIR

REQUIRED_KPI_COLUMNS = {"KPI", "District"}
REQUIRED_PRICING_COLUMNS = {
    "timestamp",
    "electricity_pricing-$/kWh",
    "electricity_pricing_predicted_1-$/kWh",
    "electricity_pricing_predicted_2-$/kWh",
    "electricity_pricing_predicted_3-$/kWh",
    "carbon_intensity-kgCO2/kWh",
}
REQUIRED_BUILDING_COLUMNS = {
    "timestamp",
    "Energy Production from PV-kWh",
    "Net Electricity Consumption-kWh",
    "Cooling Demand-kWh",
    "DHW Demand-kWh",
    "Indoor Dry Bulb Temperature-C",
    "Occupant Count",
    "Power Outage",
}
REQUIRED_BATTERY_COLUMNS = {
    "timestamp",
    "Battery Soc-%",
    "Battery (Dis)Charge-kWh",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_simulation_dir(simulation_dir: Path) -> dict[str, object]:
    if simulation_dir.parent.name != "SimulationData":
        raise ValueError(
            f"{simulation_dir} is not directly under a SimulationData root"
        )

    grouped_files: dict[str, dict[str, Path]] = {}
    for file_path in sorted(simulation_dir.iterdir()):
        if not file_path.is_file():
            continue
        webkit_relative_path = (
            f"{simulation_dir.parent.name}/{simulation_dir.name}/{file_path.name}"
        )
        path_parts = webkit_relative_path.split("/")
        if len(path_parts) != 3:
            raise ValueError(f"unexpected upload path shape: {webkit_relative_path}")

        grouped_files.setdefault(path_parts[1], {})[path_parts[2]] = file_path

    file_map = grouped_files.get(simulation_dir.name)
    if not file_map:
        raise ValueError(f"official UI parser would not discover {simulation_dir.name}")

    if "exported_kpis.csv" not in file_map:
        raise ValueError("missing exported_kpis.csv")

    kpi_rows = _read_csv_rows(file_map["exported_kpis.csv"])
    if not kpi_rows:
        raise ValueError("exported_kpis.csv is empty")
    if not REQUIRED_KPI_COLUMNS <= set(kpi_rows[0].keys()):
        raise ValueError("exported_kpis.csv is missing required KPI columns")

    parsed_data_files: dict[str, list[dict[str, str]]] = {}
    available_episodes: set[str] = set()
    for file_name, file_path in file_map.items():
        if re.search(r"kpi(s)?", file_name, re.IGNORECASE):
            continue

        cleaned_name = file_name.replace("exported_data_", "").rsplit(".", 1)[0]
        match = re.search(r"_?(ep\d+)", cleaned_name, re.IGNORECASE)
        if match:
            available_episodes.add(match.group(1))

        parsed_data_files[cleaned_name] = _read_csv_rows(file_path)

    if "pricing_ep0" not in parsed_data_files:
        raise ValueError("missing exported_data_pricing_ep0.csv")
    if not parsed_data_files["pricing_ep0"]:
        raise ValueError("pricing export is empty")
    if not REQUIRED_PRICING_COLUMNS <= set(parsed_data_files["pricing_ep0"][0].keys()):
        raise ValueError("pricing export is missing required columns")

    building_keys = sorted(
        key for key in parsed_data_files if re.fullmatch(r"building_\d+_ep\d+", key)
    )
    battery_keys = sorted(
        key for key in parsed_data_files if re.fullmatch(r"building_\d+_battery_ep\d+", key)
    )
    if not building_keys:
        raise ValueError("no building exports discovered")
    if not battery_keys:
        raise ValueError("no battery exports discovered")

    if not REQUIRED_BUILDING_COLUMNS <= set(parsed_data_files[building_keys[0]][0].keys()):
        raise ValueError("building export is missing required columns")
    if not REQUIRED_BATTERY_COLUMNS <= set(parsed_data_files[battery_keys[0]][0].keys()):
        raise ValueError("battery export is missing required columns")

    return {
        "simulation": simulation_dir.name,
        "episodes": sorted(available_episodes),
        "kpi_rows": len(kpi_rows),
        "building_files": building_keys,
        "battery_files": battery_keys,
        "pricing_file": "pricing_ep0",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "simulation_root",
        nargs="?",
        default=str(RESULTS_DIR / "ui_exports" / "SimulationData"),
        help="path to the SimulationData root or one simulation directory",
    )
    args = parser.parse_args()

    candidate = Path(args.simulation_root).resolve()
    simulation_dirs: list[Path]
    if candidate.name == "SimulationData":
        simulation_dirs = sorted(path for path in candidate.iterdir() if path.is_dir())
    else:
        simulation_dirs = [candidate]

    if not simulation_dirs:
        raise SystemExit("no simulation directories found")

    payload = {
        "root": str(candidate),
        "validated_simulations": [_validate_simulation_dir(path) for path in simulation_dirs],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

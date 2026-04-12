from __future__ import annotations

import math
from typing import Any


def _normalize_value(value: Any) -> float | None:
    if value is None:
        return None

    number = float(value)
    if math.isnan(number):
        return None

    return number


def district_kpis(env: Any) -> dict[str, float | None]:
    frame = env.evaluate()
    district_rows = frame[frame["level"] == "district"]
    return {
        row["cost_function"]: _normalize_value(row["value"])
        for row in district_rows.to_dict("records")
    }


def challenge_metrics(env: Any) -> dict[str, dict[str, float | str | None]]:
    metrics = env.evaluate_citylearn_challenge()
    payload = {}

    for key, value in metrics.items():
        payload[key] = {
            "display_name": value["display_name"],
            "weight": value["weight"],
            "value": _normalize_value(value["value"]),
        }

    return payload


def build_metrics_payload(
    env: Any,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    challenge = challenge_metrics(env)
    raw = district_kpis(env)
    return {
        **run_context,
        "average_score": challenge["average_score"]["value"],
        "challenge_metrics": challenge,
        "district_kpis": raw,
    }


def flatten_metrics_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": payload["run_id"],
        "algorithm": payload["algorithm"],
        "variant": payload["variant"],
        "split": payload["split"],
        "seed": payload["seed"],
        "dataset_name": payload["dataset_name"],
        "average_score": payload["average_score"],
    }

    for key, value in payload["challenge_metrics"].items():
        row[key] = value["value"]

    for key, value in payload["district_kpis"].items():
        row[f"district_{key}"] = value

    return row

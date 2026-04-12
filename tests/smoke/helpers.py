import os
from pathlib import Path

import pytest

from cos435_citylearn.dataset import DEFAULT_DATASET_NAME
from cos435_citylearn.paths import DATA_DIR


def require_benchmark_runtime() -> None:
    try:
        import citylearn  # noqa: F401
    except ImportError:
        if os.getenv("COS435_REQUIRE_DATA") == "1":
            pytest.fail(
                "CityLearn benchmark runtime is not installed. run `make install-benchmark`"
            )

        pytest.skip("CityLearn benchmark runtime is not installed")


def require_dataset() -> Path:
    schema_path = (
        DATA_DIR
        / "external"
        / "citylearn_2023"
        / DEFAULT_DATASET_NAME
        / "schema.json"
    )

    if not schema_path.exists():
        if os.getenv("COS435_REQUIRE_DATA") == "1":
            pytest.fail("CityLearn 2023 dataset is missing. run `make download-citylearn`")

        pytest.skip("CityLearn 2023 dataset is missing")

    manifest_path = DATA_DIR / "manifests" / "citylearn_2023_manifest.json"
    if not manifest_path.exists() and os.getenv("COS435_REQUIRE_DATA") == "1":
        pytest.fail("CityLearn 2023 dataset manifest is missing")

    return schema_path

from pathlib import Path

import yaml

from cos435_citylearn.paths import CONFIGS_DIR


def test_all_yaml_configs_parse() -> None:
    config_paths = sorted(CONFIGS_DIR.rglob("*.yaml"))
    assert config_paths

    for path in config_paths:
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), f"{path} did not parse into a mapping"


def test_train_configs_have_shared_top_level_shape() -> None:
    required = {"env", "algorithm", "reward", "features", "training", "evaluation", "logging"}
    train_paths = sorted((CONFIGS_DIR / "train").rglob("*.yaml"))

    assert train_paths

    for path in train_paths:
        data = yaml.safe_load(path.read_text())
        assert required.issubset(data.keys()), f"{Path(path).name} is missing train sections"

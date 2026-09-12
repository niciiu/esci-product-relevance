"""YAML configuration loading utilities."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config file into a plain dict.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the file does not parse into a top-level mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a top-level mapping: {path}"
        )

    return config

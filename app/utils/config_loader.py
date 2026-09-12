"""
config_loader.py

Centralized YAML configuration loader for the Contract Intelligence Platform.
All modules must read settings via this loader instead of hardcoding values
or re-implementing YAML parsing. Results are cached so files are read once
per process.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels up from this file: app/utils/config_loader.py -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"

SETTINGS_PATH = _CONFIG_DIR / "settings.yaml"
RETRIEVAL_PATH = _CONFIG_DIR / "retrieval.yaml"
EXTRACTION_KEYWORDS_PATH = _CONFIG_DIR / "extraction_keywords.yaml"


class ConfigError(Exception):
    """Raised when a configuration file is missing or malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file from disk and return its contents as a dictionary.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed YAML contents as a dictionary.

    Raises:
        ConfigError: If the file does not exist or fails to parse.
    """
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} did not parse to a dictionary.")
    return data


@functools.lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    """
    Load and cache config/settings.yaml.

    Returns:
        Dictionary of general application settings.
    """
    return _load_yaml(SETTINGS_PATH)


@functools.lru_cache(maxsize=1)
def get_retrieval_config() -> dict[str, Any]:
    """
    Load and cache config/retrieval.yaml.

    Returns:
        Dictionary of retrieval/chunking hyperparameters.
    """
    return _load_yaml(RETRIEVAL_PATH)


@functools.lru_cache(maxsize=1)
def get_extraction_keywords() -> dict[str, Any]:
    """
    Load and cache config/extraction_keywords.yaml.

    Returns:
        Dictionary mapping field_type names to lists of keyword/phrase patterns.
    """
    return _load_yaml(EXTRACTION_KEYWORDS_PATH)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT

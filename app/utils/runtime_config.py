"""Read runtime values from process environment or Streamlit secrets."""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)


def _streamlit_secret(name: str) -> str | None:
    """Return a configured Streamlit secret without exposing access errors."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        # Streamlit raises when secrets are not configured, including in local
        # command-line use. Absence should preserve the existing YAML defaults.
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def get_runtime_setting(name: str) -> str | None:
    """Resolve an environment variable first, then a Streamlit secret."""
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    return _streamlit_secret(name)


def get_runtime_bool(name: str, default: bool) -> bool:
    """Resolve a boolean setting from env/Streamlit secrets, else use default."""
    value = get_runtime_setting(name)
    if value is None:
        return default

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    _logger.warning("Ignoring invalid %s setting; using configured default.", name)
    return default

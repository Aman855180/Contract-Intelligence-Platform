"""
file_storage.py

Helper for persisting uploaded file bytes (e.g. from a Streamlit file
uploader) to the configured raw_documents_dir before ingestion. Kept
separate from document_parser.py so that ingest_document() can also be
called directly against files already on disk (e.g. in tests or scripts).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.utils.config_loader import get_project_root, get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def save_uploaded_file(file_bytes: bytes, original_filename: str) -> Path:
    """
    Persist raw uploaded file bytes to disk under paths.raw_documents_dir,
    prefixing with a short random token to avoid filename collisions
    between concurrent uploads of files with the same name.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        original_filename: The original filename (used to preserve extension
            and for a human-readable prefix).

    Returns:
        Path to the saved file on disk.
    """
    settings = get_settings()
    raw_dir_rel = settings["paths"]["raw_documents_dir"]
    raw_dir: Path = get_project_root() / raw_dir_rel
    raw_dir.mkdir(parents=True, exist_ok=True)

    original_path = Path(original_filename)
    unique_prefix = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_prefix}_{original_path.name}"
    destination = raw_dir / safe_filename

    destination.write_bytes(file_bytes)
    logger.info("Saved uploaded file to %s (%d bytes).", destination, len(file_bytes))
    return destination

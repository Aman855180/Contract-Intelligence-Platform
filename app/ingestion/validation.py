"""
validation.py

Pre-parsing validation for uploaded contract files: extension whitelist and
file size limit, both read from config/settings.yaml. Validation happens
before any parsing library is invoked, so bad input fails fast and cheaply.
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.models.document import DocumentType
from app.utils.config_loader import get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

_EXTENSION_TO_TYPE = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
}


def validate_and_classify(file_path: Path) -> DocumentType:
    """
    Validate that a file is an allowed extension and within the configured
    size limit, then return its classified DocumentType.

    Args:
        file_path: Path to the file on disk to validate.

    Returns:
        The DocumentType corresponding to the file's extension.

    Raises:
        UnsupportedFileTypeError: If the file extension is not in
            ingestion.allowed_extensions in settings.yaml.
        FileTooLargeError: If the file exceeds ingestion.max_file_size_mb.
    """
    settings = get_settings()
    ingestion_cfg = settings.get("ingestion", {})
    allowed_extensions: list[str] = ingestion_cfg.get("allowed_extensions", [".pdf", ".docx"])
    max_file_size_mb: float = ingestion_cfg.get("max_file_size_mb", 25)

    extension = file_path.suffix.lower()
    if extension not in allowed_extensions:
        logger.warning("Rejected file with unsupported extension: %s", file_path.name)
        raise UnsupportedFileTypeError(
            f"File extension '{extension}' is not supported. Allowed: {allowed_extensions}"
        )

    file_size_bytes = file_path.stat().st_size
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_file_size_bytes:
        logger.warning(
            "Rejected oversized file: %s (%.2f MB > %.2f MB limit)",
            file_path.name,
            file_size_bytes / (1024 * 1024),
            max_file_size_mb,
        )
        raise FileTooLargeError(
            f"File '{file_path.name}' is {file_size_bytes / (1024 * 1024):.2f} MB, "
            f"exceeding the {max_file_size_mb} MB limit."
        )

    return _EXTENSION_TO_TYPE[extension]

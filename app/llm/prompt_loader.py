"""
prompt_loader.py

Loads externalized prompt templates from prompts/*.txt and formats them
with runtime variables. Per project rules, no prompt text is ever
hardcoded as inline strings inside Python modules — every prompt sent to
the LLM passes through this loader.
"""

from __future__ import annotations

import functools
from pathlib import Path

from app.llm.exceptions import PromptNotFoundError
from app.utils.config_loader import get_project_root
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


@functools.lru_cache(maxsize=None)
def _read_prompt_file(prompt_name: str) -> str:
    """
    Read and cache a prompt template file's raw contents.

    Args:
        prompt_name: Filename (with .txt extension) under the prompts/ directory.

    Returns:
        The raw template text, with {placeholder} format fields intact.

    Raises:
        PromptNotFoundError: If the file does not exist under prompts/.
    """
    prompt_path: Path = get_project_root() / "prompts" / prompt_name
    if not prompt_path.exists():
        raise PromptNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_prompt(prompt_name: str, **kwargs: object) -> str:
    """
    Load a prompt template and format it with the given keyword arguments.

    Args:
        prompt_name: Filename (with .txt extension) under prompts/, e.g. "qa_prompt.txt".
        **kwargs: Values to substitute into the template's {placeholder} fields.

    Returns:
        The fully formatted prompt string ready to send to the LLM.

    Raises:
        PromptNotFoundError: If the template file does not exist.
        LLMError-derived KeyError surfaces as-is if a required placeholder is missing
            from kwargs (caught and re-raised with context for easier debugging).
    """
    template = _read_prompt_file(prompt_name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        logger.error("Missing placeholder %s when formatting prompt '%s'.", exc, prompt_name)
        raise PromptNotFoundError(
            f"Prompt '{prompt_name}' requires placeholder {exc} which was not provided."
        ) from exc

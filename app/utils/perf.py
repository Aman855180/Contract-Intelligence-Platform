"""
perf.py

High-resolution performance instrumentation using time.perf_counter().
Provides context managers and logging helpers for operation latency
and LLM call metrics.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from typing import Any, Generator

from app.utils.logging_setup import get_logger

logger = get_logger("perf")

_RECORDED_SPANS: list[dict[str, Any]] = []
_CURRENT_LLM_ATTEMPT: contextvars.ContextVar[int] = contextvars.ContextVar("current_llm_attempt", default=1)


def get_current_llm_attempt() -> int:
    """Return the current LLM attempt number in this context."""
    return _CURRENT_LLM_ATTEMPT.get()


@contextlib.contextmanager
def llm_attempt_scope(attempt: int) -> Generator[None, None, None]:
    """Context manager to set the active LLM attempt number."""
    token = _CURRENT_LLM_ATTEMPT.set(attempt)
    try:
        yield
    finally:
        _CURRENT_LLM_ATTEMPT.reset(token)


@contextlib.contextmanager
def perf_span(
    operation: str,
    log_logger: Any = None,
    extra_info: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager to measure and log operation duration using time.perf_counter().

    Logs:
        [PERF] operation=<name> duration=<seconds>s
    """
    active_logger = log_logger or logger
    start = time.perf_counter()
    active_logger.info("[PERF] operation=%s status=START%s", operation, f" {extra_info}" if extra_info else "")
    record: dict[str, Any] = {
        "operation": operation,
        "start": start,
        "duration": 0.0,
        "extra_info": extra_info,
    }
    try:
        yield record
    finally:
        duration = time.perf_counter() - start
        record["duration"] = duration
        _RECORDED_SPANS.append(record)
        active_logger.info(
            "[PERF] operation=%s duration=%.4fs%s",
            operation,
            duration,
            f" {extra_info}" if extra_info else "",
        )


def log_llm_perf(
    provider: str,
    model: str,
    duration: float,
    prompt_chars: int,
    response_chars: int,
    attempt: int = 1,
    log_logger: Any = None,
) -> None:
    """
    Log structured LLM performance block.

    Logs:
        [LLM_PERF]
        provider=<provider>
        model=<model>
        duration=<seconds>
        prompt_chars=<number>
        response_chars=<number>
        attempt=<number>
    """
    active_logger = log_logger or logger
    active_logger.info(
        "[LLM_PERF]\n"
        "provider=%s\n"
        "model=%s\n"
        "duration=%.4f\n"
        "prompt_chars=%d\n"
        "response_chars=%d\n"
        "attempt=%d",
        provider,
        model,
        duration,
        prompt_chars,
        response_chars,
        attempt,
    )


def clear_recorded_spans() -> None:
    """Clear in-memory recorded performance spans."""
    _RECORDED_SPANS.clear()


def get_recorded_spans() -> list[dict[str, Any]]:
    """Return a copy of all recorded performance spans."""
    return list(_RECORDED_SPANS)

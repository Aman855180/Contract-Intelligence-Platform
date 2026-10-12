"""In-process cold-cache versus warm-cache benchmark for the contract app.

Each invocation creates a unique, retained artifact directory under
``data/benchmark_runs``. Cold and warm phases use separate SQLite databases
and Chroma stores so Q&A can use the existing global retrieval API without
seeing unrelated contracts. Both phases ingest the same PDF and ask the same
question. No benchmark artifacts or user data are deleted.

This is a cache-cold/warm comparison inside one Python process, not a fresh
process-start benchmark. Application ML caches are reset before the cold phase
and retained for the warm phase; phase-specific Chroma and BM25 caches are
reset to isolate each phase's single benchmark document.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_QUESTION = "What is the employee's title?"
SECRET_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY")


class BenchmarkInputError(ValueError):
    """Raised when the required source document cannot be benchmarked."""


class BenchmarkRunError(RuntimeError):
    """Safe failure summary for a benchmark phase; never stores raw exception text."""

    def __init__(
        self,
        phase: str,
        original: BaseException,
        metrics: dict[str, Any],
    ) -> None:
        self.phase = phase
        self.exception_type = type(original).__name__
        self.safe_message = sanitize_error(str(original))
        self.frames = [
            f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}"
            for frame in traceback.extract_tb(original.__traceback__)[-6:]
        ]
        self.metrics = metrics
        super().__init__(f"{phase} failed ({self.exception_type}): {self.safe_message}")


def sanitize_error(message: str) -> str:
    """Redact known environment secrets and common credential formats."""
    for env_name in SECRET_ENV_VARS:
        secret = os.getenv(env_name)
        if secret:
            message = message.replace(secret, "[REDACTED_SECRET]")
    patterns = (
        (r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+", r"\1[REDACTED_SECRET]"),
        (r"(?i)([?&](?:key|api[_-]?key)=)[^&\s]+", r"\1[REDACTED_SECRET]"),
        (r"(?i)AIzaSy[A-Za-z0-9_-]+", "[REDACTED_SECRET]"),
    )
    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message)
    return message[:500]


def validate_pdf_path(value: str | Path) -> Path:
    """Fail fast unless the requested PDF exists, is readable, and is nonempty."""
    path = Path(value).expanduser()
    if not path.exists():
        raise BenchmarkInputError(f"PDF does not exist: {path}")
    if not path.is_file():
        raise BenchmarkInputError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise BenchmarkInputError("Input must have a .pdf extension.")
    try:
        with path.open("rb") as source:
            first_byte = source.read(1)
    except OSError as exc:
        raise BenchmarkInputError(
            f"PDF is not readable ({type(exc).__name__}): {sanitize_error(str(exc))}"
        ) from None
    if not first_byte:
        raise BenchmarkInputError("PDF is empty.")
    return path.resolve()


@dataclass
class MeasurementRecorder:
    """Benchmark-only observations that the application does not span itself."""

    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    gemini_calls: list[dict[str, Any]] = field(default_factory=list)
    ner_model_loading_seconds: float = 0.0
    ner_inference_seconds: float = 0.0
    qa_retrieved_chunks: int | None = None
    qa_found_in_contract: bool | None = None

    def clear_step(self) -> None:
        self.provider_calls.clear()
        self.gemini_calls.clear()
        self.ner_model_loading_seconds = 0.0
        self.ner_inference_seconds = 0.0
        self.qa_retrieved_chunks = None
        self.qa_found_in_contract = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider_calls": list(self.provider_calls),
            "gemini_calls": list(self.gemini_calls),
            "ner_model_loading_seconds": self.ner_model_loading_seconds,
            "ner_inference_seconds": self.ner_inference_seconds,
            "qa_retrieved_chunks": self.qa_retrieved_chunks,
            "qa_found_in_contract": self.qa_found_in_contract,
        }


def extract_metrics(
    spans: list[dict[str, Any]],
    observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate actual application span names without adding nested parents.

    Parent and child timings are returned separately. In particular, callers
    must not add ``hybrid_retrieval`` to its vector/BM25/fusion child spans.
    """
    observations = observations or {}
    by_operation: dict[str, float] = defaultdict(float)
    for span in spans:
        operation = str(span.get("operation", ""))
        duration = span.get("duration", 0.0)
        if operation and isinstance(duration, (int, float)) and math.isfinite(duration):
            by_operation[operation] += max(0.0, float(duration))

    embedding_load = by_operation.get("embedding_model_loading", 0.0)
    nli_load = by_operation.get("nli_model_loading", 0.0)
    ner_load = float(observations.get("ner_model_loading_seconds", 0.0))
    embedding_inference = sum(
        by_operation.get(name, 0.0) for name in ("embed_texts", "embed_queries")
    )
    nli_inference = sum(
        by_operation.get(name, 0.0)
        for name in ("nli_inference", "nli_inference_contradiction")
    )
    provider_calls = observations.get("provider_calls", [])
    gemini_calls = observations.get("gemini_calls", [])
    return {
        "span_seconds": dict(by_operation),
        "model_loading_seconds": {
            "embedding_model_loading": embedding_load,
            "nli_model_loading": nli_load,
            "benchmark_ner_model_loading": ner_load,
            "total_distinct_model_loads": embedding_load + nli_load + ner_load,
        },
        "embedding_inference_seconds": embedding_inference,
        "nli_inference_seconds": nli_inference,
        "benchmark_ner_inference_seconds": float(
            observations.get("ner_inference_seconds", 0.0)
        ),
        "retrieval_span_seconds": {
            name: by_operation.get(name, 0.0)
            for name in (
                "hybrid_retrieval",
                "risk_context_retrieval",
                "report_context_retrieval",
                "chroma_vector_retrieval",
                "chroma_vector_retrieval_batch",
                "bm25_retrieval",
                "bm25_index_build",
                "rrf_fusion",
            )
            if name in by_operation
        },
        "field_extraction_seconds": by_operation.get("field_extraction", 0.0),
        "provider_calls": provider_calls,
        "provider_generation_seconds": sum(
            float(call.get("seconds", 0.0)) for call in provider_calls
        ),
        "gemini_calls": gemini_calls,
        "gemini_api_seconds": sum(
            float(call.get("seconds", 0.0)) for call in gemini_calls
        ),
        "gemini_input_tokens": _sum_optional(gemini_calls, "input_tokens"),
        "gemini_output_tokens": _sum_optional(gemini_calls, "output_tokens"),
        "qa_retrieved_chunks": observations.get("qa_retrieved_chunks"),
        "qa_found_in_contract": observations.get("qa_found_in_contract"),
    }


def _sum_optional(items: list[dict[str, Any]], key: str) -> int | None:
    values = [item.get(key) for item in items if isinstance(item.get(key), int)]
    return sum(values) if values else None


def run_measured_step(
    name: str,
    action: Callable[[], Any],
    recorder: MeasurementRecorder,
) -> tuple[Any, dict[str, Any]]:
    """Run one operation; on error preserve safe diagnostics and never fake success."""
    from app.utils.perf import clear_recorded_spans, get_recorded_spans

    clear_recorded_spans()
    recorder.clear_step()
    started = time.perf_counter()
    try:
        result = action()
    except Exception as exc:
        wall_seconds = time.perf_counter() - started
        metrics = extract_metrics(get_recorded_spans(), recorder.snapshot())
        metrics["wall_seconds"] = wall_seconds
        raise BenchmarkRunError(name, exc, metrics) from None
    wall_seconds = time.perf_counter() - started
    metrics = extract_metrics(get_recorded_spans(), recorder.snapshot())
    metrics["wall_seconds"] = wall_seconds
    return result, metrics


class _Instrumentation:
    """Temporary in-process timers for NER and provider/SDK calls."""

    def __init__(self, recorder: MeasurementRecorder) -> None:
        self.recorder = recorder
        self._restorers: list[Callable[[], None]] = []

    def install(self) -> None:
        from app.extraction import ner_pipeline
        from app.extraction import obligation_extractor
        from app.llm.gemini_provider import GeminiProvider
        from app.llm.ollama_provider import OllamaProvider

        original_load = ner_pipeline._load_ner_pipeline
        original_extract = ner_pipeline.extract_entities

        def timed_load() -> Any:
            was_loaded = ner_pipeline._pipeline_instance is not None
            started = time.perf_counter()
            try:
                return original_load()
            finally:
                if not was_loaded:
                    self.recorder.ner_model_loading_seconds += time.perf_counter() - started

        ner_pipeline._load_ner_pipeline = timed_load

        def timed_extract(text: str) -> Any:
            load_before = self.recorder.ner_model_loading_seconds
            started = time.perf_counter()
            result = original_extract(text)
            elapsed = time.perf_counter() - started
            load_during_call = self.recorder.ner_model_loading_seconds - load_before
            self.recorder.ner_inference_seconds += max(0.0, elapsed - load_during_call)
            return result

        ner_pipeline.extract_entities = timed_extract
        obligation_extractor.extract_entities = timed_extract
        self._restorers.extend(
            [
                lambda: setattr(ner_pipeline, "_load_ner_pipeline", original_load),
                lambda: setattr(ner_pipeline, "extract_entities", original_extract),
                lambda: setattr(obligation_extractor, "extract_entities", original_extract),
            ]
        )

        for provider_class in (GeminiProvider, OllamaProvider):
            original_generate = provider_class.generate

            def timed_generate(
                instance: Any,
                *args: Any,
                _original: Callable[..., Any] = original_generate,
                **kwargs: Any,
            ) -> Any:
                started = time.perf_counter()
                call: dict[str, Any] = {
                    "provider": instance.provider_name,
                    "model": instance.model_name,
                }
                try:
                    result = _original(instance, *args, **kwargs)
                    call["status"] = "success"
                    return result
                except Exception as exc:
                    call["status"] = "failure"
                    call["exception_type"] = type(exc).__name__
                    call["safe_message"] = sanitize_error(str(exc))
                    raise
                finally:
                    call["seconds"] = time.perf_counter() - started
                    self.recorder.provider_calls.append(call)

            provider_class.generate = timed_generate
            self._restorers.append(
                lambda cls=provider_class, method=original_generate: setattr(cls, "generate", method)
            )

        try:
            from google.genai.models import Models
        except ImportError:
            return

        original_sdk_generate = Models.generate_content

        def timed_sdk_generate(instance: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            call: dict[str, Any] = {"model": kwargs.get("model", "unknown")}
            try:
                response = original_sdk_generate(instance, *args, **kwargs)
                usage = getattr(response, "usage_metadata", None)
                call.update(
                    {
                        "status": "success",
                        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                    }
                )
                return response
            except Exception as exc:
                call["status"] = "failure"
                call["exception_type"] = type(exc).__name__
                call["safe_message"] = sanitize_error(str(exc))
                call["http_status"] = _safe_http_status(exc)
                call["api_code"] = _safe_api_code(exc)
                raise
            finally:
                call["seconds"] = time.perf_counter() - started
                self.recorder.gemini_calls.append(call)

        Models.generate_content = timed_sdk_generate
        self._restorers.append(
            lambda: setattr(Models, "generate_content", original_sdk_generate)
        )

    def restore(self) -> None:
        for restore in reversed(self._restorers):
            restore()
        self._restorers.clear()


def _safe_http_status(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def _safe_api_code(exc: BaseException) -> int | str | None:
    value = getattr(exc, "code", None)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        return value
    return None


def _clear_cold_model_caches() -> None:
    """Reset model and prompt caches before the cache-cold phase."""
    from app.embeddings.embedder import reset_model_cache
    from app.extraction.ner_pipeline import reset_pipeline_cache
    from app.groundedness.nli_checker import reset_model_cache as reset_nli_model_cache
    from app.llm.prompt_loader import _read_prompt_file

    reset_model_cache()
    reset_pipeline_cache()
    reset_nli_model_cache()
    _read_prompt_file.cache_clear()


def _configure_isolated_phase(settings: dict[str, Any], phase_dir: Path) -> None:
    """Point app storage at this phase's unique, retained artifact directory."""
    settings.setdefault("paths", {})["sqlite_db_path"] = str(phase_dir / "contracts.db")
    settings["paths"]["chroma_persist_dir"] = str(phase_dir / "chroma_store")

    from app.embeddings.chroma_store import reset_client_cache
    from app.retriever.bm25_index import reset_index_cache

    reset_client_cache()
    reset_index_cache()


def _assert_isolated_index(document_id: str) -> None:
    from app.embeddings.chroma_store import get_or_create_collection

    data = get_or_create_collection().get(include=["metadatas"])
    indexed_document_ids = {
        metadata.get("document_id") for metadata in (data.get("metadatas") or [])
    }
    if indexed_document_ids != {document_id}:
        raise RuntimeError(
            "The isolated benchmark index contains unexpected document IDs."
        )


def _print_step(label: str, metrics: dict[str, Any]) -> None:
    print(f"  {label}: wall={metrics['wall_seconds']:.3f}s")
    loads = metrics["model_loading_seconds"]
    print(
        "    model loads: embedding={embedding_model_loading:.3f}s "
        "NLI={nli_model_loading:.3f}s NER={benchmark_ner_model_loading:.3f}s".format(**loads)
    )
    print(
        "    inference: embedding={:.3f}s NLI={:.3f}s NER={:.3f}s".format(
            metrics["embedding_inference_seconds"],
            metrics["nli_inference_seconds"],
            metrics["benchmark_ner_inference_seconds"],
        )
    )
    if metrics["field_extraction_seconds"]:
        print(f"    field extraction: {metrics['field_extraction_seconds']:.3f}s")
    if metrics["qa_retrieved_chunks"] is not None:
        print(f"    Q&A retrieved chunks: {metrics['qa_retrieved_chunks']}")
    if metrics["qa_found_in_contract"] is not None:
        print(f"    Q&A found_in_contract: {metrics['qa_found_in_contract']}")
    for call in metrics["provider_calls"]:
        failure_detail = (
            f" error={call.get('exception_type')}: {call.get('safe_message')}"
            if call.get("status") == "failure"
            else ""
        )
        print(
            "    provider call: {provider}/{model} {status} {seconds:.3f}s{detail}".format(
                **call, detail=failure_detail
            )
        )
    for call in metrics["gemini_calls"]:
        token_text = (
            f" input_tokens={call.get('input_tokens')} output_tokens={call.get('output_tokens')}"
            if call.get("input_tokens") is not None or call.get("output_tokens") is not None
            else " tokens=unavailable"
        )
        print(
            "    Gemini SDK request: {status} {seconds:.3f}s{tokens}".format(
                **call, tokens=token_text
            )
        )
    # Parent and child spans intentionally remain separately visible; none
    # are added together to create an aggregate that double-counts nesting.
    for operation, duration in metrics["span_seconds"].items():
        print(f"    span {operation}: {duration:.3f}s")


def _print_wall_comparison(cold: dict[str, Any], warm: dict[str, Any]) -> None:
    print("\nCold-cache vs warm-cache wall time (same process):")
    for label, key in (
        ("Ingestion + indexing", "ingestion_metrics"),
        ("Q&A", "qa_metrics"),
        ("Risk Analysis", "risk_metrics"),
        ("Report Generation", "report_metrics"),
    ):
        cold_seconds = cold[key]["wall_seconds"]
        warm_seconds = warm[key]["wall_seconds"]
        change = (
            (warm_seconds - cold_seconds) / cold_seconds * 100.0
            if cold_seconds > 0
            else None
        )
        change_text = f"{change:+.1f}%" if change is not None else "n/a"
        print(
            f"  {label}: cold={cold_seconds:.3f}s "
            f"warm={warm_seconds:.3f}s change={change_text}"
        )


def _run_phase(
    phase: str,
    phase_dir: Path,
    source_pdf: Path,
    question: str,
    settings: dict[str, Any],
    recorder: MeasurementRecorder,
) -> dict[str, Any]:
    print(f"\n--- {phase.title()} cache phase ---")
    from app.chunking.pipeline import index_document
    from app.database.repository import insert_parsed_document
    from app.database.sqlite_db import init_db
    from app.embeddings.chroma_store import collection_count
    from app.ingestion.document_parser import ingest_document
    from app.llm.qa_service import answer_question
    from app.models.document import ParsedDocument
    from app.models.report import ReportSchema
    from app.models.retrieval import QAAnswer
    from app.models.risk import RiskAssessment
    from app.reports.report_service import ReportResult, generate_contract_report
    from app.risk.risk_service import assess_risk

    _configure_isolated_phase(settings, phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=False)
    run_tag = phase_dir.parent.name
    contract_name = f"BENCHMARK_{run_tag}_{phase.upper()}"
    results: dict[str, Any] = {"document_id": None, "contract_name": contract_name}

    def measured_step(label: str, action: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
        return run_measured_step(f"{phase.title()} / {label}", action, recorder)

    def ingest_and_index() -> tuple[Any, list[Any]]:
        init_db()
        parsed = ingest_document(source_pdf, contract_name=contract_name)
        if not isinstance(parsed, ParsedDocument):
            raise TypeError("ingest_document did not return ParsedDocument")
        insert_parsed_document(parsed)
        chunks = index_document(parsed)
        if not chunks or collection_count() == 0:
            raise RuntimeError("Document indexing produced no searchable chunks")
        _assert_isolated_index(parsed.metadata.document_id)
        results["document_id"] = parsed.metadata.document_id
        return parsed, chunks

    ingested, ingestion_metrics = measured_step("ingestion", ingest_and_index)
    results["ingestion_metrics"] = ingestion_metrics
    document_id = ingested[0].metadata.document_id
    print(f"  artifact document: {contract_name} ({document_id})")
    _print_step("ingestion + indexing", ingestion_metrics)

    # answer_question has no document_id parameter. The phase-specific Chroma
    # store and BM25 corpus contain exactly this document, and the wrapper
    # asserts that all returned chunks belong to it.
    from app.llm import qa_service

    original_hybrid_retrieve = qa_service.hybrid_retrieve

    def benchmark_only_retrieve(query: str) -> Any:
        retrieval = original_hybrid_retrieve(query)
        unexpected = {
            chunk.document_id
            for chunk in retrieval.chunks
            if chunk.document_id != document_id
        }
        if unexpected:
            raise RuntimeError(
                "Q&A retrieval escaped the phase-isolated benchmark document."
            )
        recorder.qa_retrieved_chunks = len(retrieval.chunks)
        return retrieval

    qa_service.hybrid_retrieve = benchmark_only_retrieve
    try:
        def run_qa() -> Any:
            result = answer_question(question)
            if not isinstance(result, QAAnswer):
                raise RuntimeError("answer_question did not return a QAAnswer")
            recorder.qa_found_in_contract = result.found_in_contract
            return result

        qa_answer, qa_metrics = measured_step("Q&A", run_qa)
    finally:
        qa_service.hybrid_retrieve = original_hybrid_retrieve
    results["qa_metrics"] = qa_metrics
    _print_step("Q&A", qa_metrics)

    def run_risk() -> Any:
        result = assess_risk(document_id)
        if not isinstance(result, RiskAssessment):
            raise RuntimeError("assess_risk did not return RiskAssessment")
        return RiskAssessment.model_validate(result.model_dump())

    risk_result, risk_metrics = measured_step("Risk Analysis", run_risk)
    results["risk_metrics"] = risk_metrics
    _print_step("Risk Analysis", risk_metrics)

    def run_report() -> Any:
        result = generate_contract_report(document_id)
        if not isinstance(result, ReportResult) or not isinstance(result.report, ReportSchema):
            raise RuntimeError("generate_contract_report did not return a validated ReportResult")
        ReportSchema.model_validate(result.report.model_dump())
        return result

    report_result, report_metrics = measured_step("Report Generation", run_report)
    results["report_metrics"] = report_metrics
    _print_step("Report Generation", report_metrics)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="One readable PDF used in both phases")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Same Q&A question for cold and warm")
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "data" / "benchmark_runs"),
        help="Root for retained isolated SQLite/Chroma artifacts (default: data/benchmark_runs)",
    )
    return parser


def run_benchmark(pdf_value: str, question: str, artifacts_dir: str) -> int:
    source_pdf = validate_pdf_path(pdf_value)
    if not question.strip():
        raise BenchmarkInputError("Question cannot be empty.")

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    from app.utils.config_loader import get_settings

    settings = get_settings()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    artifact_root = Path(artifacts_dir).expanduser().resolve() / run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    print(f"Source PDF: {source_pdf}")
    print(f"Question (identical in both phases): {question}")
    print(f"Retained isolated artifacts: {artifact_root}")
    print(
        "Mode: one-process cache-cold then warm; model caches reset before cold and retained for warm. "
        "This is not a fresh-process benchmark. Phase-local Chroma/BM25 are reset for document isolation. "
        "Workflow provider factories receive explicit settings and bypass their instance cache; "
        "Gemini creates its SDK client per request."
    )

    from app.llm.provider import (
        get_qa_provider,
        get_report_provider,
        get_risk_provider,
    )
    from app.utils.perf import clear_recorded_spans

    _clear_cold_model_caches()
    # Resolve with custom settings objects so provider names can be reported
    # without pre-populating the application's provider-instance cache.
    selected = {
        "Q&A": get_qa_provider(settings),
        "Risk": get_risk_provider(settings),
        "Report": get_report_provider(settings),
    }
    for workflow, provider in selected.items():
        print(f"Configured provider: {workflow}={provider.provider_name}/{provider.model_name}")

    recorder = MeasurementRecorder()
    instrumentation = _Instrumentation(recorder)
    instrumentation.install()
    clear_recorded_spans()
    try:
        cold = _run_phase(
            "cold", artifact_root / "cold", source_pdf, question, settings, recorder
        )
        warm = _run_phase(
            "warm", artifact_root / "warm", source_pdf, question, settings, recorder
        )
    finally:
        instrumentation.restore()
    _print_wall_comparison(cold, warm)
    print("\nBENCHMARK=SUCCESS")
    print(f"Run ID: {run_id}")
    print(f"Cold document ID: {cold['document_id']}")
    print(f"Warm document ID: {warm['document_id']}")
    print(f"Artifacts retained at: {artifact_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_benchmark(args.pdf, args.question, args.artifacts_dir)
    except BenchmarkInputError as exc:
        print(f"BENCHMARK=NOT_STARTED error={type(exc).__name__}: {sanitize_error(str(exc))}", file=sys.stderr)
        return 2
    except BenchmarkRunError as exc:
        print(
            f"BENCHMARK=FAILED phase={exc.phase} error={exc.exception_type}: {exc.safe_message}",
            file=sys.stderr,
        )
        _print_step(f"{exc.phase} partial measurements", exc.metrics)
        for frame in exc.frames:
            print(f"  at {frame}", file=sys.stderr)
        for call in exc.metrics.get("gemini_calls", []):
            print(
                "  Gemini SDK diagnostic: {status} {exception_type} "
                "HTTP={http_status} API_CODE={api_code} {seconds:.3f}s {safe_message}".format(
                    status=call.get("status"),
                    exception_type=call.get("exception_type", ""),
                    http_status=call.get("http_status", "unavailable"),
                    api_code=call.get("api_code", "unavailable"),
                    seconds=call.get("seconds", 0.0),
                    safe_message=call.get("safe_message", ""),
                ),
                file=sys.stderr,
            )
        return 1
    except Exception as exc:
        print(
            f"BENCHMARK=FAILED phase=setup error={type(exc).__name__}: {sanitize_error(str(exc))}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

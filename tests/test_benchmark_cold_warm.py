from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.benchmark_cold_warm import (
    BenchmarkInputError,
    BenchmarkRunError,
    MeasurementRecorder,
    build_parser,
    extract_metrics,
    run_measured_step,
    validate_pdf_path,
)


def test_benchmark_requires_one_pdf_argument() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_missing_pdf_fails_before_benchmark_setup() -> None:
    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        with pytest.raises(BenchmarkInputError, match="does not exist"):
            validate_pdf_path(Path(temp_dir) / "missing.pdf")


def test_pdf_must_be_readable_nonempty_pdf() -> None:
    with TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        temp_path = Path(temp_dir)
        directory = temp_path / "contract.pdf"
        directory.mkdir()
        with pytest.raises(BenchmarkInputError, match="not a file"):
            validate_pdf_path(directory)

        empty_pdf = temp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        with pytest.raises(BenchmarkInputError, match="empty"):
            validate_pdf_path(empty_pdf)

        non_pdf = temp_path / "contract.txt"
        non_pdf.write_text("readable", encoding="utf-8")
        with pytest.raises(BenchmarkInputError, match=".pdf extension"):
            validate_pdf_path(non_pdf)


def test_metric_extraction_uses_actual_spans_without_summing_nested_retrieval() -> None:
    metrics = extract_metrics(
        [
            {"operation": "embedding_model_loading", "duration": 2.0},
            {"operation": "embed_queries", "duration": 0.4},
            {"operation": "hybrid_retrieval", "duration": 3.0},
            {"operation": "chroma_vector_retrieval", "duration": 0.8},
            {"operation": "bm25_retrieval", "duration": 0.2},
            {"operation": "nli_model_loading", "duration": 1.0},
            {"operation": "nli_inference", "duration": 0.3},
            {"operation": "field_extraction", "duration": 1.5},
        ],
        {
            "ner_model_loading_seconds": 0.5,
            "ner_inference_seconds": 0.25,
            "gemini_calls": [
                {"seconds": 1.2, "input_tokens": 100, "output_tokens": 12}
            ],
        },
    )

    assert metrics["model_loading_seconds"] == {
        "embedding_model_loading": 2.0,
        "nli_model_loading": 1.0,
        "benchmark_ner_model_loading": 0.5,
        "total_distinct_model_loads": 3.5,
    }
    assert metrics["embedding_inference_seconds"] == 0.4
    assert metrics["nli_inference_seconds"] == 0.3
    assert metrics["benchmark_ner_inference_seconds"] == 0.25
    assert metrics["field_extraction_seconds"] == 1.5
    assert metrics["retrieval_span_seconds"] == {
        "hybrid_retrieval": 3.0,
        "chroma_vector_retrieval": 0.8,
        "bm25_retrieval": 0.2,
    }
    assert metrics["gemini_api_seconds"] == 1.2
    assert metrics["gemini_input_tokens"] == 100
    assert metrics["gemini_output_tokens"] == 12


def test_failed_phase_is_not_reported_as_success_and_redacts_credentials(monkeypatch) -> None:
    secret = "benchmark-secret-value"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    def fail() -> None:
        raise RuntimeError(f"request failed Authorization: Bearer {secret}")

    with pytest.raises(BenchmarkRunError) as error:
        run_measured_step("Q&A", fail, MeasurementRecorder())

    assert error.value.phase == "Q&A"
    assert error.value.exception_type == "RuntimeError"
    assert secret not in error.value.safe_message
    assert "[REDACTED_SECRET]" in error.value.safe_message
    assert error.value.metrics["wall_seconds"] >= 0

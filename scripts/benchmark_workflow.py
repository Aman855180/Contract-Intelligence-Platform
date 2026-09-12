#!/usr/bin/env python3
"""
benchmark_workflow.py

Executes a representative end-to-end Contract Intelligence workflow and
outputs a detailed latency breakdown across all major stages.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.chunking.pipeline import index_document
from app.database.repository import delete_document, insert_parsed_document
from app.database.sqlite_db import init_db
from app.embeddings.chroma_store import delete_document_chunks
from app.extraction.extraction_service import extract_fields_from_document
from app.groundedness.nli_checker import check_groundedness
from app.ingestion.document_parser import ingest_document
from app.llm.qa_service import answer_question
from app.reports.report_service import generate_contract_report
from app.risk.risk_service import assess_risk
from app.utils.logging_setup import get_logger
from app.utils.perf import clear_recorded_spans, get_recorded_spans, perf_span

logger = get_logger("benchmark")


def main() -> None:
    print("=" * 70)
    print("Starting Representative Contract Intelligence Benchmark")
    print("=" * 70)

    init_db()
    clear_recorded_spans()

    # Locate sample PDF contract
    sample_pdf = _REPO_ROOT / "data" / "raw_documents" / "3977760c_Employment_Agreement_V1.pdf"
    if not sample_pdf.exists():
        # Fall back to any PDF found in data/raw_documents
        pdfs = list((_REPO_ROOT / "data" / "raw_documents").glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError("No sample PDF contracts found in data/raw_documents.")
        sample_pdf = pdfs[0]

    print(f"Contract file: {sample_pdf.name} ({sample_pdf.stat().st_size} bytes)")

    contract_name = "Employment_Agreement_Benchmark"

    with perf_span("contract_analysis_e2e_workflow", extra_info=f"file={sample_pdf.name}"):
        # Stage 1: Document Loading & Parsing
        print("\n--- 1. Document Ingestion ---")
        parsed_doc = ingest_document(sample_pdf, contract_name=contract_name)
        doc_id = parsed_doc.metadata.document_id

        try:
            delete_document_chunks(doc_id)
            delete_document(doc_id)
        except Exception:
            pass

        insert_parsed_document(parsed_doc)

        # Stage 2: Chunking, Embedding, and Chroma Indexing
        print("\n--- 2. Document Chunking & Vector Indexing ---")
        chunks = index_document(parsed_doc)
        print(f"Indexed {len(chunks)} chunks.")

        # Stage 3: Structured Field Extraction
        print("\n--- 3. Structured Field Extraction ---")
        extraction = extract_fields_from_document(doc_id)
        print(f"Extracted {len(extraction.fields)} fields.")

        # Stage 4: Risk Assessment (Rule-based + LLM + Scoring)
        print("\n--- 4. Risk Assessment ---")
        assessment = assess_risk(doc_id, extraction=extraction)
        print(f"Risk assessment complete. Score: {assessment.risk_score}/100, Triggered flags: {len(assessment.triggered_flags())}")

        # Stage 5: Executive Report Generation (Context retrieval + LLM JSON + Markdown)
        print("\n--- 5. Executive Report Generation ---")
        report_res = generate_contract_report(doc_id, extraction=extraction, assessment=assessment)
        print(f"Report generated: {len(report_res.report.key_parties)} parties, {len(report_res.report.key_clauses)} clauses. Markdown size: {len(report_res.markdown)} chars.")

        # Stage 6: Grounded Contract Q&A (Hybrid retrieval + LLM + NLI)
        print("\n--- 6. Grounded Contract Q&A ---")
        question = "What are the termination conditions and notice period?"
        qa_res = answer_question(question)
        print(f"Question: {question}")
        print(f"Answer: {qa_res.answer[:150]}...")
        if qa_res.groundedness:
            print(f"Groundedness score: {qa_res.groundedness.entailment_score:.4f} (grounded={qa_res.groundedness.is_grounded})")

    # Print summary table
    spans = get_recorded_spans()
    print("\n" + "=" * 70)
    print(f"{'OPERATION':<38} | {'DURATION (s)':<12} | {'DETAILS'}")
    print("-" * 70)
    for span in spans:
        op = span.get("operation", "")
        dur = span.get("duration", 0.0)
        extra = span.get("extra_info") or ""
        print(f"{op:<38} | {dur:>10.4f}s | {extra}")
    print("=" * 70)


if __name__ == "__main__":
    main()

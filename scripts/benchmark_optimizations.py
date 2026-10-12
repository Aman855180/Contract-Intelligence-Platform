import os
import sys
import time
import pathlib

sys.path.insert(0, os.getcwd())

from app.ingestion.document_parser import ingest_document
from app.chunking.pipeline import index_document
from app.llm.qa_service import answer_question
from app.risk.risk_service import assess_risk
from app.reports.report_service import generate_contract_report
from app.database.sqlite_db import init_db

def run_benchmark():
    print("Initializing...")
    init_db()

    import glob
    pdfs = glob.glob("data/raw_documents/*Employment_Agreement_V2.pdf")
    contract_path = pdfs[0]

    # Ingest
    t0 = time.perf_counter()
    parsed_doc = ingest_document(pathlib.Path(contract_path), "Opt_Bench_Contract")
    chunks = index_document(parsed_doc)
    doc_id = parsed_doc.metadata.document_id
    print(f"Ingested {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s")

    for i in range(3):
        print(f"\n--- RUN {i+1} ---")
        # Q&A
        t0 = time.perf_counter()
        qa_res = answer_question("What is the employee's title?")
        print(f"Q&A took {time.perf_counter() - t0:.2f}s. Answer: {qa_res.answer[:50]}...")

        # Risk
        try:
            t0 = time.perf_counter()
            risk_res = assess_risk(doc_id)
            print(f"Risk took {time.perf_counter() - t0:.2f}s. Score: {risk_res.risk_score}")
        except Exception as e:
            print(f"Risk failed: {e}")

        # Report
        try:
            t0 = time.perf_counter()
            rep_res = generate_contract_report(doc_id)
            print(f"Report took {time.perf_counter() - t0:.2f}s. Clauses: {len(rep_res.key_clauses)}, Recs: {len(rep_res.recommendations)}")
        except Exception as e:
            print(f"Report failed: {e}")

if __name__ == "__main__":
    run_benchmark()

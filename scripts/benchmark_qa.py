import os
import sys
import time

sys.path.insert(0, os.getcwd())

from app.database.sqlite_db import init_db
from app.llm.qa_service import answer_question
from app.utils.perf import get_recorded_spans, clear_recorded_spans

def main():
    init_db()
    clear_recorded_spans()

    # Pre-load models to ensure clean latency (as done in production for cold starts)
    from app.embeddings.embedder import _load_model as load_embedder
    load_embedder()

    from app.groundedness.nli_checker import _load_nli_model
    _load_nli_model()

    question = "what is the salary in agreement 2"

    print(f"\nAsking question: '{question}'...")

    start_time = time.perf_counter()
    answer = answer_question(question)
    total_latency = time.perf_counter() - start_time

    print("\n--- Answer ---")
    print(f"Answer: {answer.answer}")
    print(f"Groundedness: {answer.groundedness}")
    print(f"Source: {answer.source_contract} page {answer.source_page}")

    print("\n--- Latency Breakdown ---")
    spans = get_recorded_spans()

    retrieval_latency = 0.0
    embedding_latency = 0.0
    llm_latency = 0.0
    nli_latency = 0.0
    provider_used = "unknown"
    fallback_occurred = False

    for span in spans:
        op = span.get("operation")
        dur = span.get("duration", 0)

        if op == "hybrid_retrieval_query":
            retrieval_latency += dur
        elif op == "embed_queries":
            embedding_latency += dur
        elif op == "llm_request_end":
            llm_latency += dur
            provider_used = span.get("provider", "unknown")
        elif op == "llm_request_failed":
            fallback_occurred = True
        elif op == "groundedness_check":
            nli_latency += dur

        print(f"{op}: {dur:.4f}s")

    print("\n--- Summary ---")
    print(f"Retrieval Latency: {retrieval_latency:.4f}s")
    print(f"Embedding Latency: {embedding_latency:.4f}s")
    print(f"LLM Latency: {llm_latency:.4f}s")
    print(f"NLI Latency: {nli_latency:.4f}s")
    print(f"Total Q&A Latency: {total_latency:.4f}s")
    print(f"Provider Used: {provider_used}")
    print(f"Fallback Occurred: {fallback_occurred}")

if __name__ == '__main__':
    main()

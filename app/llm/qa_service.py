"""
qa_service.py

Grounded contract Q&A: retrieves relevant chunks via hybrid retrieval,
asks the LLM to answer strictly from those chunks with a citation, and
independently verifies the answer's groundedness via the NLI model rather
than trusting the LLM's own claim. This is the single entry point Phase
11's Streamlit UI calls for the Q&A feature.
"""

from __future__ import annotations

from app.groundedness.nli_checker import check_groundedness
from app.llm.ollama_client import generate
from app.llm.prompt_loader import load_prompt
from app.llm.response_parsers import parse_qa_response
from app.models.retrieval import QAAnswer, RetrievedChunk
from app.retriever.hybrid_retriever import hybrid_retrieve
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """
    Render retrieved chunks into the context block injected into qa_prompt.txt.

    Args:
        chunks: The chunks returned by hybrid_retrieve(), already top_k limited.

    Returns:
        A formatted string with one block per chunk, each labeled with its
        contract name and page number so the LLM can cite accurately.
    """
    blocks = []
    for chunk in chunks:
        blocks.append(f"[{chunk.contract_name}, page {chunk.page_number}]\n{chunk.text}")
    return "\n\n".join(blocks)


def _find_source_chunk(
    chunks: list[RetrievedChunk], contract_name: str | None, page: int | None
) -> RetrievedChunk | None:
    """
    Locate the retrieved chunk matching the LLM-cited contract name and page,
    used as the groundedness check's reference passage. Falls back to the
    top-ranked chunk if no exact match is found (e.g. minor name formatting
    differences from the LLM).

    Args:
        chunks: The retrieved chunks the LLM was given as context.
        contract_name: The contract name the LLM cited.
        page: The page number the LLM cited.

    Returns:
        The matching RetrievedChunk, or the first chunk as a fallback, or
        None if chunks is empty.
    """
    if not chunks:
        return None
    for chunk in chunks:
        if chunk.contract_name == contract_name and chunk.page_number == page:
            return chunk
    logger.warning(
        "LLM cited '%s' page %s, which doesn't exactly match any retrieved chunk; "
        "falling back to top-ranked chunk for groundedness check.",
        contract_name,
        page,
    )
    return chunks[0]


def answer_question(question: str) -> QAAnswer:
    """
    Answer a user's question about the indexed contract(s), grounded in
    retrieved passages, with an independently-verified groundedness score.

    Args:
        question: The user's natural-language question.

    Returns:
        A QAAnswer. If no sufficiently relevant evidence was retrieved, or
        the LLM itself reports no answer, found_in_contract is False and no
        groundedness check is performed.
    """
    retrieval = hybrid_retrieve(question)

    if not retrieval.above_threshold or not retrieval.chunks:
        logger.info("No sufficiently relevant evidence retrieved for question: '%s'", question)
        return QAAnswer(question=question, answer="NOT_FOUND_IN_CONTRACT", found_in_contract=False)

    retrieved_context = _format_retrieved_context(retrieval.chunks)
    prompt = load_prompt("qa_prompt.txt", retrieved_context=retrieved_context, question=question)

    raw_response = generate(prompt)
    parsed = parse_qa_response(raw_response)

    if parsed["answer"] == "NOT_FOUND_IN_CONTRACT":
        return QAAnswer(question=question, answer="NOT_FOUND_IN_CONTRACT", found_in_contract=False)

    source_contract = parsed["source_contract"]
    source_page = parsed["source_page"]
    supporting_text = parsed["supporting_text"] or ""

    source_chunk = _find_source_chunk(retrieval.chunks, source_contract, source_page)
    groundedness_reference_text = supporting_text if supporting_text else (source_chunk.text if source_chunk else "")

    groundedness = check_groundedness(claim=parsed["answer"], source_passage=groundedness_reference_text)

    return QAAnswer(
        question=question,
        answer=parsed["answer"],
        found_in_contract=True,
        source_contract=source_contract or (source_chunk.contract_name if source_chunk else None),
        source_page=source_page or (source_chunk.page_number if source_chunk else None),
        supporting_text=supporting_text or None,
        groundedness=groundedness,
    )

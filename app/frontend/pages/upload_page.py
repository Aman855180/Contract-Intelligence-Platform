"""
upload_page.py

Upload & Manage page: accepts PDF/DOCX files, runs ingestion + chunking +
indexing, and shows all contracts indexed in this session.
"""

from __future__ import annotations

import streamlit as st

from app.chunking.pipeline import index_document
from app.database.repository import delete_document, insert_parsed_document
from app.embeddings.chroma_store import delete_document_chunks
from app.frontend.state import add_document, get_documents, remove_document
from app.ingestion.document_parser import ingest_document
from app.ingestion.file_storage import save_uploaded_file
from app.models.document import DocumentType


def render() -> None:
    st.header("📂 Upload & Manage Contracts")
    st.write(
        "Upload PDF or DOCX contracts. Each file is ingested, chunked, and "
        "indexed for retrieval within this session."
    )

    # ── File uploader ────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Choose one or more contract files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Text-based PDFs and DOCX files only. Scanned/image-only PDFs are not supported.",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Skip files already indexed in this session (by original name).
            already_indexed_names = {
                meta.contract_name for meta in get_documents().values()
            }
            base_name = uploaded_file.name.rsplit(".", 1)[0]
            if base_name in already_indexed_names:
                st.info(f"**{uploaded_file.name}** is already indexed. Skipping.")
                continue

            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                try:
                    file_bytes = uploaded_file.read()
                    file_path = save_uploaded_file(file_bytes, uploaded_file.name)
                    parsed = ingest_document(file_path, contract_name=base_name)
                    insert_parsed_document(parsed)
                    index_document(parsed)
                    add_document(parsed.metadata)
                    st.success(
                        f"✓ **{parsed.metadata.contract_name}** — "
                        f"{parsed.metadata.page_count} page(s), "
                        f"{parsed.metadata.document_type.value.upper()}"
                    )
                except Exception as exc:
                    st.error(f"Failed to process **{uploaded_file.name}**: {exc}")

    st.divider()

    # ── Indexed contracts table ──────────────────────────────────────────────
    docs = get_documents()
    if not docs:
        st.info("No contracts indexed yet. Upload files above to get started.")
        return

    st.subheader(f"Indexed Contracts ({len(docs)})")

    for doc_id, meta in list(docs.items()):
        col_name, col_type, col_pages, col_del = st.columns([4, 1, 1, 1])
        with col_name:
            st.write(f"**{meta.contract_name}**")
        with col_type:
            st.write(meta.document_type.value.upper())
        with col_pages:
            st.write(f"{meta.page_count} p.")
        with col_del:
            if st.button("🗑", key=f"del_{doc_id}", help="Remove from session"):
                try:
                    delete_document_chunks(doc_id)
                    delete_document(doc_id)
                except Exception:
                    pass  # already removed or not in SQLite
                remove_document(doc_id)
                st.rerun()

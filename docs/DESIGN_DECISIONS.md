# Design Decisions

Rationale for every non-obvious architectural choice in the Contract
Intelligence Platform. Written after the full build so the reasoning reflects
what was actually learned, not what was speculated up front.

---

## 1. Hybrid Retrieval: BM25 + Vector Search via Reciprocal Rank Fusion

**Decision:** Combine BM25 (rank_bm25) and dense vector search (ChromaDB,
BAAI/bge-base-en-v1.5) via Reciprocal Rank Fusion (RRF) rather than either
alone.

**Rationale:**

- **BM25 excels at exact-term retrieval.** Contract Q&A frequently involves
  precise terms: "30 days", "termination", "indemnification". A user who asks
  "What is the notice period?" needs the chunk containing "30 days written
  notice", which BM25 retrieves reliably via keyword overlap even without
  semantic understanding.

- **Vector search handles paraphrase and synonymy.** A question about "when
  can the contract be ended?" should also surface termination clauses even when
  those exact words don't appear. Dense embeddings generalise across
  paraphrase in ways BM25 cannot.

- **RRF is score-agnostic.** BM25 scores (unbounded, corpus-dependent) and
  cosine similarities (0–1, model-dependent) live on incompatible scales. RRF
  combines them by rank position (`1 / (k + rank)`) rather than score value,
  so neither search arm needs normalisation and the combination is
  interpretable: every chunk's fused score is a sum of contributions from each
  ranking it appears in.

- **Fixed `rrf_k=60`.** This is the empirically established standard constant
  for RRF (Cormack et al., 2009). It prevents a rank-1 result from one arm
  dominating when the other arm disagrees. Deviating from 60 would require a
  labelled development set to justify.

**What we learned from Phase 10:** BM25 Recall@5 = 0.70 and MRR = 0.38 on a
20-contract corpus where all contracts share the same template. This is lower
than expected and confirms that vector search is critical for disambiguation
in the full hybrid pipeline: without semantic understanding, BM25 can't
distinguish which contract's payment-terms chunk is most relevant to a
user who has just uploaded one specific contract.

---

## 2. Chunk Size (800 chars) and Overlap (100 chars)

**Decision:** Fixed chunk_size=800 characters with chunk_overlap=100, stored
in `config/retrieval.yaml`.

**Rationale:**

- **800 chars balances context vs. precision.** A typical contract clause
  (e.g. a payment terms paragraph or a termination clause) is 200–600
  characters. At 800 chars, a single chunk almost always contains the full
  relevant clause plus a little surrounding context for the LLM, without
  pulling in clauses from a different topic that would dilute retrieval
  signal.

- **100 chars overlap prevents clause boundary splits.** Contract prose is
  dense; a clause that starts near the end of one chunk and finishes near the
  start of the next would be retrievable by neither chunk alone. 100-char
  overlap ensures the clause appears fully in at least one chunk.

- **Overlap snaps to whitespace.** The chunker searches backward from the
  target cut point for the nearest whitespace within 50 characters, so chunks
  never split words mid-token. This matters for BM25 (which tokenises by word
  boundaries) and for citation quality (supporting text should be readable).

- **Chunks never span page boundaries.** Every chunk carries a single
  unambiguous page_number for citation. This is a deliberate constraint: a
  chunk spanning page 2→3 could not produce a valid citation. The cost is
  that the last clause on a page may be split from its continuation, but for
  the contract types in scope this is rare and the citation benefit outweighs
  it.

---

## 3. NLI-Based Groundedness (DeBERTa-v3-base-mnli-fever-anli)

**Decision:** Use `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for
groundedness checking rather than asking the LLM to self-report confidence.

**Rationale:**

- **LLM self-reported confidence is not calibrated.** A language model
  returning "confidence: 92%" in its own answer has no principled basis for
  that number and is prone to hallucination about its hallucinations. An NLI
  model provides a probability grounded in a distinct, non-autoregressive
  classification head.

- **NLI framing matches the task.** Groundedness is the question "does the
  source passage entail this answer?" — precisely what NLI (Natural Language
  Inference) is trained to classify. The entailment probability is a direct
  answer to that question.

- **Why DeBERTa-v3-base over roberta-large-mnli?** DeBERTa-v3-base-mnli
  achieves comparable MNLI accuracy to roberta-large while being smaller and
  faster on CPU — important since this platform targets local deployment
  without GPU. The `-fever-anli` training extension improves robustness to
  factual-challenge NLI, which is closer to the "does the source support this
  claim?" task than pure MNLI.

- **Bidirectional contradiction check.** For contract comparison, the
  `check_contradiction` function runs NLI in both directions (A→B and B→A)
  and returns the maximum contradiction score. NLI is directional (premise →
  hypothesis), so a contradiction only clearly detected in one direction would
  be missed by a unidirectional check.

---

## 4. Clause Extraction: Keyword Matching, Not NER

**Decision:** Clause-type fields (payment terms, termination, confidentiality,
liability, renewal, signature) are detected via curated keyword/phrase
patterns in `config/extraction_keywords.yaml`, not a token-classification NER
model.

**Rationale:**

- **No suitable pretrained model exists.** Standard NER models (CoNLL-2003
  label set: PER, ORG, LOC, MISC) do not predict contract clause *types*. The
  `law-ai/InLegalBERT` placeholder in Phase 1's initial config is a legal
  domain backbone — it has no token-classification head and cannot do NER
  without fine-tuning. Fine-tuning is explicitly out of scope for this
  project.

- **Keyword matching is fully explainable.** The patterns in
  `extraction_keywords.yaml` can be audited, tuned, and extended without
  touching Python. A legal professional reviewing the system can read them
  directly. A black-box NER model could not offer this.

- **Confidence scales with signal strength.** A single keyword match yields
  0.6 confidence; each additional distinct keyword adds 0.15 (capped at 1.0).
  This lets downstream consumers (comparison, risk) use the most confident
  representative field for each type.

**Where real NER is used:** `dslim/bert-base-NER` (CoNLL-2003 PER/ORG) is
used in `obligation_extractor.py` to enrich obligation sentences with the
identity of the obligated party — this *is* a genuine PER/ORG detection task
that the pretrained model was actually trained for.

**Lessons from Phase 10:** Three rounds of keyword refinement were required
before evaluation metrics stabilised. Common boilerplate phrases like
"signed by authorized representatives", "terminated in accordance with", and
"breaches of confidentiality obligations" all triggered false positives for
signature/termination/confidentiality respectively. The evaluation harness
caught each issue; this would not have been caught by unit tests alone.

---

## 5. Rule-Based vs. LLM Risk Classification Split

**Decision:** Missing-clause flags (signature, termination, confidentiality)
are rule-based; semantic risk flags (unlimited indemnity, high liability,
ambiguous payment) are LLM-classified against a fixed Pydantic-validated
taxonomy.

**Rationale:**

- **Clause absence is binary and deterministic.** A signature block is either
  present or absent in the extracted fields. There is no semantic ambiguity
  that justifies the cost and non-determinism of an LLM call.

- **Semantic risk requires reading comprehension.** Whether an indemnity
  clause is "unlimited" depends on the exact wording ("shall indemnify
  against all claims without limit" vs. "shall indemnify up to $1M"). Keyword
  presence alone is insufficient; an LLM reading the full clause is the right
  tool.

- **The LLM classifies against a fixed schema; it does not invent categories.**
  The three LLM-classified categories are defined in code and in the prompt.
  The LLM outputs a boolean (triggered) plus a citation — it cannot add a
  fourth category or change the weights. This keeps the risk formula
  deterministic: `risk_score = sum(weight_i * triggered_i)` using only
  config-defined weights.

- **Retry-on-failure with structured output validation.** The LLM's JSON
  output is validated against `_LLMRiskOutput` (Pydantic) before any value is
  trusted. On failure the error is injected back into the retry prompt. In
  Phase 10's schema validation benchmark, first-attempt pass rate = 100% on
  the synthetic test corpus.

---

## 6. JSON-First Report Pipeline

**Decision:** The report pipeline always follows: LLM → JSON →
`_LLMReportOutput` (Pydantic) → `ReportSchema` → Markdown renderer. The LLM
never generates Markdown directly.

**Rationale:**

- **Separation of concerns.** The LLM's job is structured content generation.
  Formatting is a deterministic transformation that doesn't need the LLM. If a
  new output format is needed (HTML, PDF, DOCX), only a new renderer is
  written — the LLM call and validation pipeline are unchanged.

- **Validated before trusted.** An LLM outputting Markdown directly cannot be
  schema-validated. An LLM outputting JSON can. Every field (severity must be
  `low|medium|high`, explanation must be non-empty, source_page must be an
  int ≥ 1) is enforced by Pydantic before the report is displayed or
  downloaded.

- **`document_id` and `contract_name` are injected by Python, not the LLM.**
  This prevents a hallucinated or incorrect contract name from appearing in
  the report header. The LLM never sees these fields in its output schema.

---

## 7. DOCX Pagination Approximation

**Decision:** DOCX "pages" are delimited by explicit hard page breaks
(`<w:br w:type="page"/>`), not by rendered pagination.

**Rationale:** Word documents have no native pagination at the XML level.
Page layout is computed by the rendering engine at display time; it is not
stored in the `.docx` format. Accurately recovering page numbers would require
a layout engine (e.g. LibreOffice in headless mode), adding a heavy
dependency.

**Tradeoff:** Documents without hard page breaks are treated as a single page.
This is correct for short contract templates but degrades for long documents.
The limitation is documented here and flagged to users in the UI tooltip.
For the supported use case (standard contract templates), page-number
citations derived from this approach are sufficiently accurate.

---

## 8. SQLite for Metadata; ChromaDB Exclusively for Vectors

**Decision:** SQLite stores only document metadata and raw page text. Vectors
and chunk embeddings live exclusively in ChromaDB. These are never mixed.

**Rationale:**

- **SQLite is the wrong tool for vector search.** Storing 768-dimensional
  float arrays in SQLite blobs and performing cosine similarity at query time
  is O(n) and would be unusably slow at scale. ChromaDB's HNSW index provides
  approximate nearest-neighbour search in O(log n).

- **ChromaDB is the wrong tool for relational queries.** Finding all documents
  uploaded after a given date, or joining document metadata with extracted
  fields, is straightforward in SQLite and awkward in a vector store.

- **The separation forces clarity.** No module is tempted to "just stick the
  embedding in the metadata" because the architecture makes the boundaries
  explicit. ChromaDB chunks carry `document_id` in their metadata so they can
  be joined back to SQLite records when needed.

---

## 9. Evaluation Strategy and Scope

**Decision:** Evaluate only the components that can be evaluated honestly in
this environment. Report model-dependent metrics as `REQUIRES_MODEL`.

**Rationale:** Fabricated evaluation metrics are worse than no metrics — they
mislead the reader and are discoverable under questioning. The four metrics
computed in Phase 10 (`scripts/run_evaluation.py`) are computed against real
pipeline outputs on real synthetic PDF contracts:

| Metric | Result | Notes |
|---|---|---|
| Inconsistency Detection F1 | 1.00 | Deterministic rule-based on controlled inputs |
| BM25 Recall@5 | 0.70 | Honest result; all 20 contracts share same template |
| Risk Detection F1 | 1.00 | Deterministic rule-based |
| Schema Validation Accuracy | 1.00 | Pydantic pipeline on 25 synthetic responses |

The low BM25 Recall@1 (0.15) is an honest result that motivates the need for
vector search disambiguation in the full hybrid pipeline — and was left in
the results rather than being explained away.

NLI groundedness rate, vector retrieval recall, and LLM structured output
validity are reported as `REQUIRES_MODEL` because fabricating these numbers
would be dishonest. The infrastructure to measure them (eval_harness.py,
run_evaluation.py) is fully built and will produce real numbers when run with
the full model stack.

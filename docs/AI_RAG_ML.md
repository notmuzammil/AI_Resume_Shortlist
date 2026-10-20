# AI, embeddings, RAG and ML — implementation details

## 1. Deterministic baseline

The original pipeline contains a curated skill taxonomy, direct aliases, directional inference, date-sensitive work history, degree extraction, weighted scoring and requirement-level evidence. It remains the default; no GPU or online API is required to evaluate a CV.

**Qualification repair:** `modules/pipeline.py` now differentiates completed degrees from degrees explicitly in progress when a completed bachelor's qualification is required. Unknown-year degrees are surfaced for human checking. The existing hard-gate rejection is *provisional human-review information*, not an instruction to deny employment.

## 2. Generative AI integrations

### Requirement verification

Existing `modules/llm_engine.py` verifies only unmatched conceptual requirements. Model output must use a structured schema and actual CV line evidence. Cloud transfers are blocked unless `LLMOptions.cloud_consent=True`, and the Streamlit interface has an explicit checkbox. Masking removes many contacts/demographic labels but is not guaranteed to anonymize all natural-language identity references.

### Structured fact proposal

`modules/structured_ai.py` optionally calls **local** Ollama `/api/chat`, asking for proposed role/skill/degree/project/certification facts tagged with numbered source lines. Pydantic validates the schema. Facts are accepted in the *proposal* only if each fact's value is a literal substring of its cited CV line. No extracted proposal automatically changes a legal degree completion gate, employment dates or score.

Set `OLLAMA_STRUCTURED_MODEL` to a locally available model name to enable; unset it to disable. API: `POST /candidates/{id}/structured-proposal`.

### JD quality review

`GET /jobs/{id}/quality-review` detects some ambiguous cases: unknown title, absence of extractable requirements, completed-degree assumption, large mandatory skill lists, and implausible fresher vs long experience combinations. It is **heuristic**, not a generative rewrite engine. Approval is always explicit.

## 3. Embedding-based semantic retrieval

Set `EMBEDDING_BACKEND=ollama` and `EMBEDDING_MODEL=<installed model>`; the API calls the local `/api/embed` route on `OLLAMA_HOST`. A second opt-in backend, `sentence-transformers`, loads a locally cached pretrained SentenceTransformer; install the additional package yourself and pre-download a model in a controlled environment. Without a model, only BM25 is used, and API responses state `mode: bm25`.

For each redacted section-aware CV chunk, a numeric embedding and model identifier are persisted. Query vector cosine similarity is fused with BM25 (reciprocal rank fusion, rank constant 60). This is **not** a trained in-house candidate ranker; the vector model is pretrained and use depends on model licensing and evaluation.

**Performance scope:** non-indexed fallback loads up to 5,000 tenant chunks for application-layer scoring. For larger tenants enable Postgres pgvector indexing (see Deployment) plus native Postgres full-text retrieval, partitioning, and per-tenant quotas. `candidate_vectors` supports optional pgvector persistence/retrieval; the baseline BM25 scan remains limited. Do not claim production-scale indexing until those operational changes are made.

## 4. Retrieval-Augmented Generation

RAG normally includes retrieval + generative synthesis. This release deliberately uses **extractive RAG / retrieval-grounded Q&A**: retrieve authorized chunks and compile citation-tagged source sentences. The generator does not call a free-form LLM, so it cannot invent a new narrative. It answers with candidate/document/chunk/source-line IDs. This has narrower expressive ability but strong evidence traceability. If no supporting chunk is found, it explicitly abstains.

**Example:** `What evidence of real PostgreSQL production work is documented?` → numbered excerpts of relevant CV text, not a definitive assertion about candidate competence.

## 5. Traditional ML readiness

**Not implemented:** fine-tuned binary suitability classifier, learning-to-rank model, employability prediction, protected-class inference. There is no recruiter-labeled dataset that would justify training. The bundled prior CV files are examples of documents, not objective hiring outcome labels.

**Implemented:** standardized retrieval metrics utility (`scripts/evaluate_retrieval.py`) that accepts independent labeled judgments and reports precision@k, recall@k, MRR and NDCG@k. For more rigorous evaluation, add evidence correctness, degree-completion accuracy, relevant-years error, fairness/error slices where permitted, group drift checks and privacy leakage audits.

### Training readiness roadmap

1. Freeze ruleset/prompt/embed-model and anonymize adjudicated rubric labels from consensual test participants.
2. Acquire representative technical + nontechnical, multilingual, complex-layout and adversarial documents.
3. Independently label requirement evidence with human reviewers; resolve disagreements by adjudication.
4. Split by candidate and job family (prevent near-duplicate train/test leakage).
5. Compare deterministic scores, BM25, embeddings and reranking on holdout sets.
6. Only then assess calibrated interpretable learning-to-rank under legal and fairness governance.
7. Monitor false rejections, ranking drift, exposure bias, reviewer overrides and source citation precision.

### Input format for evaluator

```json
[
  {"query":"backend Python API", "relevant_ids":["candidateA"], "ranked_ids":["candidateB","candidateA"]}
]
```

Run `python scripts/evaluate_retrieval.py judgments.json 10`. The example IDs are illustrative, not labels or metrics from a real evaluated dataset.

## 6. Security and fairness controls

- No cloud LLM submission unless the explicit consent flag is true.
- Demographic and identity masking before LLM and embedding use; known limitations disclosed.
- Tenant-scoped vector and keyword retrieval.
- No candidate rejection, offer or outreach automatically issued from AI scores.
- Formal human decisions are persisted with reviewer identity and written reasons.
- Model/provenance/version information accompanies match records.
- Consider EU AI Act Annex III, applicable privacy laws and independent legal review when targeting regulated recruitment.

## 7. Known model validation limitations

Local Ollama and hosted LLM end-to-end behavior cannot be verified without running the chosen model/provider. Structured parser prompts need model-specific evaluation for JSON validity, sensitivity leakage and adversarial injections. RAG citation tests exercise the extractive fallback. Embedding tests verify vector maths and fail-safe lexical fallback, not relevance against a human-labeled search corpus.

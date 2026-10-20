# Test and verification record

**Release evaluated:** AI Resume Parser enhanced v2, 15 October 2026.

## Verified runs

### A. Original functional tests using private source fixtures

Command (local private working copy only):

```bash
APP_SECRET=<private testing secret> python -m pytest tests/ --ignore=tests/test_app.py -q --disable-warnings --maxfail=5
```

**Result: 307 passed, 3 skipped, 0 failed** in 244.65 seconds. The original sample PDFs were present for this run. Legacy tests that expected a candidate with an unfinished bachelor's credential to satisfy the bachelor's gate were updated to reflect the corrected eligibility interpretation. This run intentionally excluded `test_app.py` because Streamlit was not installed in the inspection environment.

### B. Sanitized shareable source, after removing original private CVs

Command:

```bash
APP_SECRET=<private testing secret> python -m pytest tests/ -q --disable-warnings --maxfail=1
```

**Result: 248 passed, 44 skipped, 0 failed** in 23.91 seconds before the final one-test role-permission addition. Skips correspond to private source PDF fixtures deliberately excluded from the distributable ZIP; unavailable Streamlit browser/runtime tests; and optional OCR/model integrations. The release-specific new end-to-end API tests subsequently passed **16/16**, including the new reviewer role test.

**Final packaged-tree regression result:** See `TEST_SUMMARY.txt` at project root for the last complete pass after the final source edits.

### C. Syntax validation

`python -m compileall -q api modules app.py app_llm.py platform_ui.py scripts` passed. Python imports of `api.main` and schema creation passed when `APP_SECRET` was configured.

### D. Configuration validation

The Docker Compose YAML parses with a YAML parser. **Docker image build was not executed**; this is not a service-level integration check.

## Newly tested workflows

- Setup-token bootstrap and first admin, tenant JWT authentication and unauthenticated rejection.
- Organization-isolated candidate/job read, search, RAG, deletion and matching.
- Recruiter-role prohibition against admin erasure and audit access; reviewer-role write restrictions.
- JD creation, confirmation, interpretation and job quality checks.
- CV import, within-tenant deduplication, authorization-reference requirement and bad-file rejection.
- Encryption/decryption of extracted CV, encrypted object storage, post-erasure absence.
- Basic deterministic candidate profile and source-backed match with reviewer decisions.
- Persistent job applications and stage changes.
- BM25 candidate search and extractive RAG citation paths, including no-evidence abstention.
- Embedding provider outage fallback to lexical retrieval.
- Candidate fact correction recording without silent algorithm override.
- Local structured AI proposal schema and source-line evidence rejection with mocked Ollama.
- Cloud-LLM usage blocked without explicit consent (mocked check).
- Retention cleanup dry-run and actual deletion.
- Persistent matching task queue processed with a worker.
- Retrieval metric computations and vector cosine sanity.

## Not validated in this environment

- Live Streamlit browser/E2E behavior (`app.py`, `app_llm.py`, or `platform_ui.py`).
- Actual Docker build and production PostgreSQL or MinIO connections (development uses SQLite/encrypted local object store).
- pgvector extension, production vector indexes, dimension-specific ANN recall/latency.
- Actual Ollama/remote model output with production configuration.
- OCR engine / ONNX models on scanned PDFs where dependencies/models unavailable.
- Correctness of semantic similarity against independently labeled recruiter search pairs.
- Production throughput, backups/recovery, high availability, rate limits, penetration test, model fairness or jurisdictional compliance.

## Validation commands

```bash
python -m compileall -q api modules app.py app_llm.py platform_ui.py scripts
pytest -q tests/test_new_platform.py
pytest -q tests
python -m scripts.retention_cleanup             # dry-run; needs DB + APP_SECRET
python scripts/evaluate_retrieval.py labels.json 10
```

## Release criteria before real-company deployment

- Pass all unit tests and synthetic API E2E.
- Stand up Postgres/MinIO/Ollama in staging with known pinned versions and run fresh integration tests.
- Add tested, repeatable schema migrations, tenant database RLS and isolated ingestion jobs.
- Verify actual resumes from consented test participants, including manual evaluation of degree and employment parsing and sensitive-data masking.
- Human-approved benchmark for candidate relevance and evidence correctness; bias and legal review for hiring use case.
- External security test, credential rotation, durable backups and restore testing.

**Interpretation:** A passing test suite shows coverage of implemented behaviors, not a claim of certified fitness for high-risk hiring automation.

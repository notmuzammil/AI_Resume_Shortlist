# Implementation record and remaining gaps

## Scope and version

Release name: **AI Resume Parser — Recruitment Intelligence Platform v2 (developer preview)**.

Built as modifications and additions to the existing Python application. Original parser and matching modules remain in place; no destructive rewrite of the core algorithm.

## Implemented changes

### Reliability
- Hard-gate fix: explicitly unfinished bachelor-only qualification does not count as completed when the JD requires a completed bachelor's degree.
- Uncertain completion dates now generate reviewer warnings.
- Controlled fallback: unavailable embedding service leaves BM25 search available.
- Raw CV errors are returned as per-file import errors rather than crashing batch operation.

### Security and privacy
- Removed embedded secret from distributable source archive; provided `.env.example` without live credentials.
- Added explicit cloud verification consent flag; denied model call on missing consent.
- Improved inline masking for gender/nationality/address/other sensitive personal information, contact values and links.
- Added user JWT sessions, Argon2, tenant and role authorization, encrypted raw objects and parsed text, audit metadata, consent-reference capture and retention dates.
- Excluded original real CV fixtures from final shared ZIP.

### Product/backend
- Added SQLAlchemy models and persistent jobs, candidates, applications, match runs, corrections, human reviews, audit events, processing tasks.
- Added REST endpoints with tenant scoping and automatic Swagger docs.
- Added `platform_ui.py`, a basic authenticated Streamlit recruiter UI for jobs, upload, matching, evidence search, corrections and reviews.
- Added job version and recruiter confirmation workflow.
- Added candidate import, deduplication, profile, search, deletion, manual correction and review workflows.
- Added durable DB-backed matching queue and CLI worker.
- Added operational retention cleanup CLI.

### AI / RAG / ML
- Added optional source-validated local Ollama structured fact proposals.
- Added section-aware, source-line indexed chunks and cautiously verified page mapping for some PDFs.
- Added BM25 lexical retrieval and optional local pretrained embedding/vector search with RRF fusion.
- Added optional PostgreSQL pgvector table and vector lookup support.
- Added extractive retrieval-grounded assistant with explicit source citations and no hallucinated knowledge synthesis.
- Added templated interview questions and heuristic JD quality checks.
- Added offline evaluation script (precision, recall, MRR and NDCG) requiring external labels.

### Deployment and verification
- Added API dependency files, Dockerfile, MinIO + PostgreSQL Compose sample, strict secrets example, documentation and new synthetic end-to-end tests.
- Kept legacy Streamlit entrypoint for single-user workflow.

## Partial and unimplemented capabilities

| Gap | Current status | Planned work |
|---|---|---|
| Next.js multi-user recruiter frontend | Not implemented | Build secure frontend against API, with permissions and reviewer UX |
| Basic API-backed recruiter UI | Implemented, not browser tested | Improve pagination, UI/UX, browser E2E tests; legacy local UI remains separate |
| Production PostgreSQL migrations/RLS | Not implemented | Alembic migrations, policies, integration tests |
| Scalable semantic search | Partial | pgvector HNSW/IVFFlat indexes, PostgreSQL FTS, query plans, partitioning |
| Pretrained cross-encoder reranker | Not implemented | Add only after labeled reranking evaluation |
| Full generative RAG synthesis | Deliberately excluded | Keep extractive mode; optionally validate source-backed generative answers |
| Custom trained ML suitability classifier | Not appropriate yet | Independently labeled, audited dataset and fairness review needed |
| Durable async CV import/embedding | Synchronous import | Move to isolated queue workers with retries and progress |
| Document coordinate-level citations | Partial | OCR bounding boxes/layout anchoring and stronger PDF text-offset matching |
| Biographic de-identification | Partial | Human review and NER-based PII scrub + security test dataset |
| Hiring legal compliance | Not certified | Independent legal, privacy, fairness and bias reviews |
| MFA/SSO, invites and admin UI | Not implemented | Production identity provider/SSO integration |
| Rate limiting, antivirus, sandbox | Not implemented | Edge controls and isolated, timed workers |
| Candidate correspondence, ATS integrations | Not implemented | Integrations after consent/use case validation |
| SaaS billing, usage quotas | Not implemented | Subscription and tenant quota layer |
| Production Docker end-to-end | Not executed in audit runtime | Deploy in staging, check services/OCR/models/storage/migrations |
| Live Ollama and cloud LLM testing | Not executed in audit runtime | Model/provider-specific connectivity and evidence tests |
| Representative performance/fairness metrics | Not available | Human-labeled evaluations needed |

## Release decision

The code is a stronger prototype suitable for **local development, demonstrations with synthetic or consented data, and controlled internal evaluation**. It is **not fully verified for autonomous or high-risk recruitment**. Do not automate adverse hiring decisions or publicly launch without production security and quality gates.

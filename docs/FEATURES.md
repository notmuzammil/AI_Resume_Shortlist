# Product features, modules and scope

## Feature matrix

| Area | Implemented | Notes / scope |
|---|---|---|
| PDF, DOCX, TXT extraction | Yes | Existing extraction engine preserved; unreadable images need available OCR runtime |
| Image-only PDF OCR | Optional | Existing RapidOCR/ONNX path; dependency/runtime not verified in this environment |
| JD interpretation | Yes | Parses title, experience, skills and requirements; heuristic |
| Experience + degree parsing | Yes | Explicitly unfinished bachelors cannot satisfy completed-degree gate |
| Matching and scoring | Yes | Weighted rules; source snippets; not predictive probabilities |
| Optional LLM requirement checker | Yes | Ollama or opt-in cloud; restricted to unrecognized concepts; evidence checks |
| Structured AI proposal | Optional | Locally hosted Ollama; JSON schema + exact evidence line, never auto-applied |
| Tenant users + roles | Yes | JWT; admin/recruiter/reviewer; admin-generated users |
| Jobs and revisions | Yes | JD interpretation saved on each revision, explicit recruiter confirmation |
| Candidate documents | Yes | Tenant-scoped, hash deduplication, consent reference, retention date |
| Encrypted storage | Yes | Fernet on all raw CV blobs and extracted CV text (derived from APP_SECRET) |
| MinIO/S3 | Optional | Uses boto3, bucket must be private; Compose example included |
| Application workflow | Yes | Stage + notes; no automated messaging or external calendar |
| Evidence search | Yes | BM25 keyword retrieval by tenant |
| Semantic vector search | Optional | Requires configured Ollama/sentence-transformers embedding service |
| Pgvector | Optional | SQL-backed vector insert/search and base schema; production indexes need deployment-specific model dimensions |
| RAG assistant | Yes (extractive) | Returns directly sourced excerpts and document/line citations; no generative unsupported answer |
| Interview questions | Yes (template) | Standardized verification questions from missing requirements |
| Human review and corrections | Yes | Reviewer notes and evidence-backed corrections; no silent score overwrite |
| Audit logs | Yes | Who changed what and when, no source CV bodies in audit events |
| Durable worker | Yes (basic) | DB task table, PostgreSQL SKIP LOCKED path, separate worker CLI |
| Candidate erasure | Yes | Deletes raw objects, chunk text and vectors, match/application records |
| Retention cleanup | CLI | `scripts/retention_cleanup.py` (schedule externally) |
| Benchmark metrics | Yes | `scripts/evaluate_retrieval.py` consumes external human-labeled JSON |
| Trained ML ranker | No | Requires carefully labeled and assessed data; intentionally not fabricated |
| Authenticated recruiter UI | Yes (basic) | `platform_ui.py` connects to authenticated FastAPI; not browser-tested |
| SaaS payment/subscriptions | No | Out of recruitment MVP scope |
| Production SSO, MFA, SCIM | No | Deployment governance backlog |
| CI/CD, Alembic migrations | Not full | Schema creation works for fresh databases; versioned migrations are follow-up |
| Advanced source coordinate PDF citations | Partial | Page only if uniquely verified, otherwise source document + extracted line range |

## Recommended initial user roles

- **Admin:** organization configuration, users, documents including erasure, audit review, recruiter actions.
- **Recruiter:** create/edit JD, confirm requirements, upload CV, match, shortlist or hold after review, manage application stage.
- **Reviewer:** read job and candidate evidence, add formal review and fact corrections; cannot import/delete.

## JD matching process

1. Recruiter enters job text; rules interpret title, degree conditions, experience, must-have/preferred skills.
2. Recruiter reviews flags, revises conflicts and confirms interpreted requirements.
3. CV files are imported with documented candidate processing authority.
4. Text extracted/OCR'd, parsed and stored encrypted; redacted evidence chunks created for retrieval.
5. Matching run calculates relevant experience, direct/inferred skills, gates, score and reasons.
6. Results preserve JD version and CV version and require a human shortlist/rejection decision.
7. Human adds verification notes, interview questions, application stages, and correction records.

## Limitations in heuristics

- Rules cannot establish a skill is truly possessed, only that its evidence is stated in an uploaded document.
- Unknown or incomplete dates and degrees require manual verification.
- Embedding similarity cannot substitute for mandatory certifications or verified employment dates.
- Demographic data masking is best effort; do not consider excerpt storage anonymous.

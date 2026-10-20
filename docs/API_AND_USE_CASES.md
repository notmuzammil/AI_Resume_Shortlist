# REST API and complete user workflows

All routes except `/health`, `/auth/bootstrap` and `/auth/login` require a valid JWT Bearer header. `tenant_id` is always taken from the authenticated user; IDs from other tenants produce 404. Use the interactive, generated OpenAPI at `http://127.0.0.1:8000/docs` for exact request schemas and try-it-out examples.

## API endpoints

| Method | Path | Role | Purpose |
|---|---|---|---|
| GET | `/health` | Public | Health probe (not dependency readiness) |
| POST | `/auth/bootstrap` | Setup token | One-time initial tenant/admin |
| POST | `/auth/login` | Public | JWT login; expires after 8 hours |
| GET | `/auth/me` | Any authenticated | Current user/tenant |
| POST | `/admin/users` | Admin | Create recruiter account |
| POST | `/jobs` | Admin/recruiter | Store new JD and parsed version |
| GET | `/jobs` | Any authenticated | List tenant jobs |
| GET | `/jobs/{id}` | Any authenticated | JD, version, interpretations |
| PUT | `/jobs/{id}` | Admin/recruiter | Replace JD and create new version |
| GET | `/jobs/{id}/quality-review` | Any authenticated | Heuristic ambiguity checks |
| POST | `/jobs/{id}/confirm-requirements` | Admin/recruiter | Manual confirmation |
| POST | `/candidates/import` | Admin/recruiter | Multipart CV plus consent reference and retention |
| GET | `/candidates` | Any authenticated | List candidate record identifiers |
| GET | `/candidates/{id}` | Any authenticated | Deterministic profile and uncertainty |
| DELETE | `/candidates/{id}` | Admin | Erase raw file, parsed text, vector chunks and applications |
| POST | `/candidates/{id}/corrections` | Admin/recruiter/reviewer | Evidence-backed fact correction log |
| GET | `/candidates/{id}/corrections` | Any authenticated | Review recorded corrections |
| POST | `/candidates/{id}/structured-proposal` | Admin/recruiter | Optional local Ollama fact proposal; never auto-apply |
| GET | `/candidates/search?q=...` | Any authenticated | Tenant-scoped BM25 and optional vector search |
| POST | `/assistant/query` | Any authenticated | Evidence-grounded, citation-first recruitment Q&A |
| POST | `/jobs/{id}/match-runs` | Admin/recruiter | Synchronous deterministic match |
| POST | `/jobs/{id}/match-runs/queue` | Admin/recruiter | Queue durable match task |
| GET | `/tasks/{id}` | Any authenticated | Poll task status and result identifier |
| GET | `/match-runs/{id}` | Any authenticated | Versioned scores/evidence |
| POST | `/matches/{id}/reviews` | Admin/recruiter/reviewer | Human decision/reason |
| GET | `/matches/{id}/interview-guide` | Any authenticated | Non-discriminatory verification questions |
| GET | `/jobs/{id}/applications` | Any authenticated | Job's active candidate stages |
| PATCH | `/applications/{id}` | Admin/recruiter | Update stage/notes |
| GET | `/audit-events` | Admin | Review organizational audit events |

## Use case 1: Create and approve a software engineering vacancy

Actor: Recruiter.

1. POST `/jobs` with title and 30+ character JD.
2. GET `/jobs/{id}` displays parsed requirements, experience and degree gates.
3. GET `/jobs/{id}/quality-review` highlights potential ambiguities and constraints.
4. PUT `/jobs/{id}` to correct the actual JD when required; version increments, confirmation resets.
5. POST `/jobs/{id}/confirm-requirements`; approver identity preserved in `job_versions`.
6. Additional edits invalidate confirmation; archived runs keep their source version.

Edge cases: too-short text (422), invalid tenant (404), unconfirmed requirements (matching technically allowed but explicitly provisional), non-recruiter attempting edits (403).

## Use case 2: Import confidential candidate CV

Actor: Recruiter.

1. Obtain required candidate notice/processing authorization outside the application, and record its reference.
2. Multipart POST `/candidates/import`: `file=@cv.pdf`, `consent_reference=...`, `retention_days=365`.
3. File signature/extension and size validation; PDF/DOCX/TXT extraction or rejection.
4. Encrypt original raw file and normalized extracted text. Extract masked candidate chunks, optionally embed.
5. Enforce duplicate SHA-256 within tenant. Return existing candidate id for duplicate.
6. Parse education, total experience and skill evidence for reviewer reference.

Edge cases: unreadable file (422), file too large (413), no notice reference (422), unavailable embedding backend (lexical fallback with warning), scanned images without OCR (422).

## Use case 3: Explainable shortlisting

Actor: Recruiter or reviewer.

1. POST `/jobs/{id}/match-runs` with `candidate_ids` list or omit for all candidates in tenant.
2. Calculate role-specific relevant experience, degree eligibility and requirement matches.
3. Return score index, matched/missing requirements, quoted evidence, component scores, source versions and caution flags.
4. Recruiter verifies ambiguous assertions and makes the **human** shortlist/hold/interview/reject choice with POST `/matches/{id}/reviews`.
5. Recruiter may update an application stage through PATCH `/applications/{id}`.

Edge cases: zero candidates (422), unknown candidate IDs are not matched; all mismatches and source warnings must be shown for human assessment. A model score itself must never automatically reject a candidate.

## Use case 4: Candidate intelligence and RAG

Actor: Recruiter.

1. GET `/candidates/search?q=production Python PostgreSQL`.
2. Retrieval uses authorized tenant's redacted chunks. If configured, embed query and fuse vector/lexical ranks.
3. POST `/assistant/query` with `question` and optional `candidate_id` constraint.
4. Response returns evidence excerpts with numbered citations, document/chunk IDs and extracted line ranges; PDF page only where verified.
5. No evidence → explicit abstention, not a fabricated answer.

Edge cases: cross-tenant ID (404), offline embedding model (BM25 fallback), unsupported claims are not added by an LLM narrator.

## Use case 5: Correct a questionable degree or experience claim

Actor: Human reviewer.

1. Inspect encrypted-source-derived profile (authorized user) and request corroborating evidence from candidate where appropriate.
2. POST `/candidates/{id}/corrections` including field, revised value, evidence reference and substantive explanation.
3. Correction is audit logged and visible via GET corrections. It **does not** silently rewrite the uploaded source or numeric matching score.
4. Human reviewer can override downstream shortlist via a separately audited review reason.

## Use case 6: Interview preparation

Retrieve `/matches/{id}/interview-guide`; questions target CV evidence gaps and ask for concrete examples. This endpoint does not attempt to infer ability, personality, protected class or employment outcome.

## Use case 7: Erasure and retention

Admin DELETE candidate removes document files, parsed data, indexed snippets, vector records, applications, matches, reviews and corrections (subject to database commit/backup policies). Non-PII audit metadata survives for accountability. Periodic cleanup uses `python -m scripts.retention_cleanup --apply`.

## Use case 8: Durable long-running match

POST `/jobs/{id}/match-runs/queue`, run `python -m api.worker`, then GET `/tasks/{id}`. One worker for SQLite; PostgreSQL supports `SKIP LOCKED` task claim. The worker lacks comprehensive retries/timeouts; configure production controls before public use.

## External integrations not included

No SMTP/email/SMS outreach, calendar invitations, external ATS sync, automated offers or user billing. They require separate contracts, consent checks and delivery validation, and are documented as follow-ups instead of misleading placeholder endpoints.

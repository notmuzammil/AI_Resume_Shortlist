# Deployment, security, privacy and operations

## Required environment configuration

| Key | Purpose | Required |
|---|---|---|
| `APP_SECRET` | JWT signing and current Fernet document-key derivation; **strong, stable, 32+ characters** | Always |
| `SETUP_TOKEN` | One-time initial admin; remove after first setup | Initial setup only |
| `DATABASE_URL` | SQLite by default; PostgreSQL URL strongly recommended for multiple users | Production |
| `PRIVATE_STORAGE_ROOT` | Local private encrypted objects | If no S3 |
| `S3_BUCKET`, `S3_ENDPOINT` | Private MinIO or S3 bucket | If using MinIO |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | Object store access | If using MinIO |
| `OLLAMA_HOST` | Local inference server endpoint | Optional AI |
| `EMBEDDING_BACKEND`, `EMBEDDING_MODEL` | Optional local embeddings | Optional AI |
| `OLLAMA_STRUCTURED_MODEL` | Optional structured extraction proposals | Optional AI |
| `PGVECTOR_ENABLED` | Create pgvector table and use PG-backed vector ranking | Optional Postgres |
| `CORS_ORIGINS` | Comma-separated permitted frontend origins | External frontend |
| `MAX_CV_BYTES` | Upload cap; 8388608 default | Optional |

**Do not include** a real `.env`, Streamlit secrets file, JWT, private keys or uploaded CVs in a distributable ZIP. The original upload included a cloud LLM API key: revoke/rotate that key and audit the original Git history.

## Recommended local development

1. Generate your own `APP_SECRET` and `SETUP_TOKEN` with `secrets.token_urlsafe` (separately).
2. Run `pip install -r requirements-api.txt` then `uvicorn api.main:app --host 127.0.0.1 --port 8000`.
3. POST `/auth/bootstrap` once and remove `SETUP_TOKEN`.
4. Keep `.private_uploads` outside web-served paths and back up encryption keys securely.
5. Use `/docs` for authenticated development testing.
6. Run one `python -m api.worker` process if using queued match runs.

## Docker path

The `docker-compose.yml` file defines an API, one worker, PostgreSQL with pgvector, and MinIO. Set required variables by copying `.env.example` to `.env` and replacing every placeholder with strong, random values. `docker compose up --build` binds the API and object-console to loopback only. It **does not** configure an external HTTPS proxy, runtime credential manager or long-term backups.

### PostgreSQL pgvector

`PGVECTOR_ENABLED=true` initializes `CREATE EXTENSION vector` and a `candidate_vectors` table when supported. An ANN index should be created for your selected embedding dimensionality and workload, for example using pgvector HNSW on a fixed-size `embedding::vector(N)` expression, with a model-specific filter. `N` must equal your selected model output dimension. Set only after validating query explain plans and tenant filtering; the base implementation does not invent a dimension or create a universal ANN index.

All generated vectors should be treated as sensitive derived candidate data. The candidate erasure service deletes Postgres candidate vectors before removing document chunks.

### MinIO

Objects are Fernet ciphertext **before** upload. Buckets must remain private and not be exposed by public ACL. Keep server credentials separate from browser/frontend credentials; the frontend only uses authenticated API endpoints. Configure object version lifecycle and purge deleted versions in production.

## Authentication and data protection

- Each user has an `admin`, `recruiter` or `reviewer` role.
- Passwords use Argon2. JWTs expire after eight hours; user disablement is checked each API request.
- All job/candidate/notes/chunk searches filter by tenant ID; this is **application enforcement**, not RLS.
- Raw original CV binaries and normalized extracted CV text are Fernet-encrypted at rest, using a stable key derived from `APP_SECRET`. Changing `APP_SECRET` without re-encrypting will make historical documents inaccessible.
- Searchable snippets are best-effort redacted, **not anonymized**. Their controlled storage remains sensitive.
- API writes action metadata to audit log; candidate document content is excluded.
- Candidate deletion includes raw object, derived chunks, human corrections, matches, reviews and applications, with a non-PII audit marker retained.
- Retention cleanup is implemented as CLI; schedule it separately in trusted infrastructure.

## Production hardening required (NOT shipped as verified)

1. Add PostgreSQL RLS, tenant-scope security tests against SQL injection, and a dedicated migration framework (e.g. Alembic).
2. Separate document encryption from JWT signing; introduce KMS + key rotation, per-tenant key policies and encrypted backups.
3. Add TLS termination, WAF/reverse-proxy upload caps, global API rate limiting, brute-force/credential stuffing mitigation and SIEM alerts.
4. Implement MFA/SSO, refresh-token rotation/revocation, user invitation, session management, password resets and granular document access policy.
5. Add an antivirus/malware scan and ZIP-bomb defense for file uploads, sandbox PDF/OCR parsing, processing time/memory quotas and antivirus updates.
6. Convert request-path extraction and embedding to a proper isolated worker queue. The initial worker processes match runs only; ingestion remains synchronous.
7. Add retry/dead-letter queue, task TTL/reaping for crashed workers, distributed tracing, logs, metrics and alerting.
8. Define legal basis, geographic hosting, consent/notice, access logs, data-subject requests, retention, backup erasure and processor agreements per jurisdiction.
9. Penetration test across role/tenant boundaries. Current test suite is an assurance starting point, not a certification.
10. Red-team model prompts and evaluate extraction hallucination/citation quality against representative human-reviewed data.
11. Validate all third-party dependency pins/builds; the provided Dockerfile is **not an executed production build**.
12. Add candidate access/appeal channels and a human-controlled oversight workflow appropriate to applicable hiring AI regulation.

## Retention cleanup

Dry run: `python -m scripts.retention_cleanup`; delete expired candidates with `python -m scripts.retention_cleanup --apply`. Set cron or a scheduled task. Backups, replicas, object-store version histories and external LLM provider retention require separate deletion policies.

## Secure release packaging

The distributable release intentionally omits the old `.git`, Windows `.venv`, live `.streamlit/secrets.toml`, real-person sample CVs, production documents and local databases. Development tests using the original private CV PDFs should be run in a private evaluation environment.

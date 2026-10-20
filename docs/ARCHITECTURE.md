# Architecture and data flow

## Logical components

```text
                   Local Streamlit batch UI (legacy stand-alone)
                       | independent original pipeline
PDF / DOCX / TXT ---->| extractor -> experience/education -> matcher -> scorer
                       |                          |
                       |             optional evidence-constrained LLM
                       v
               [Authenticated FastAPI /docs]
                       |
            user JWT, org ID, role authorization
                       |
       +---------------+-----------------+
       |               |                 |
  Jobs / Versions   Candidates        Applications
       |               |                 |
       +--------- Match Runs ------------+
                       |
          Human Reviews / Corrections / Audit
                       |
           Postgres / SQLite database
               /                 \
   Fernet-encrypted CV text      Redacted source chunks
            |                    /                 \
    Encrypted raw files       BM25 search       Embeddings (opt-in)
     (MinIO or local)                \             /
                                   Hybrid rank (RRF)
                                          |
                               Extractive grounded RAG
                                  citation refs + lines
```

## Retained original modules

`modules/extractor.py`: file validation, PDF/docx/txt text extraction, optional OCR.
`modules/jd_profile.py`: title and requirement parsing.
`modules/skill_extractor.py`: skill taxonomy and aliases.
`modules/matcher.py`: direct and directional evidence matching.
`modules/education.py`: completed/in-progress degrees.
`modules/experience.py`: overlapping employment periods and relevant work.
`modules/scoring.py`: weighted heuristic index.
`modules/pipeline.py`: batch parsing, hard gates, explanation and optional LLM.
`modules/llm_engine.py`: local and remote verification with bounded output, evidence checks.
`modules/structured_ai.py`: optional local-only structured fact proposals.

## New backend modules

`api/main.py`: HTTP contracts, authorization and reviewer workflows.
`api/security.py`: JWT, Argon2, tenant access, Fernet authenticated encryption.
`api/database.py`: SQLAlchemy tables, indexes, initial DDL.
`api/storage.py`: encrypted local/MinIO object store.
`api/services.py`: import, score, audit, deletion, job versioning.
`api/intelligence.py`: redaction-aware chunks, optional embedding, BM25 + vector fusion, citation-first answers.
`api/worker.py`: persistent queue worker.

## Important boundaries

1. Candidate raw files and original extracted text are **encrypted**. Redacted chunk snippets remain searchable and must be treated as potentially identifying sensitive personal data.
2. Tenant ID comes from a verified JWT / database user, not an arbitrary request parameter.
3. SQL query conditions include tenant_id, and object keys are partitioned by tenant. This is application-level isolation, **not PostgreSQL RLS**. Production high-assurance SaaS should add independent row-level security and cross-tenant penetration testing.
4. A mismatch is *absence of evidence in the supplied CV*, not lack of competence.
5. Job versions and ruleset version are recorded with match runs; document versions recorded with matches.
6. Original Streamlit UI does not yet use the new API's tenant control and is only intended for local/internal access.
7. Different model/chunk versions require explicit reindexing. Vector-backed search degrades to lexical when the local model is down.

## Database entities

`tenants`, `users`, `jobs`, `job_versions`, `candidates`, `documents`, `chunks`, `applications`, `match_runs`, `matches`, `reviews`, `fact_corrections`, `audit_events`, `processing_jobs`; optional `candidate_vectors` under PostgreSQL with pgvector.

## Retrieval design

- Split around resume section headings and bounded line ranges; no unsupported page number assertions.
- Mask common demographic/contact fields and remove a short first-line candidate name.
- Use BM25 if no embedding provider; use configured local pretrained embeddings if available.
- For hybrid search, combine BM25 rank and vector rank using reciprocal rank fusion.
- Evaluate against authorized tenant-chunk set and return candidate/document/chunk citations.
- Generator is deliberately extractive (not a hallucination-prone LLM narrative). Recruiter questions return directly quoted evidence.

## Operational notes

The demonstration worker uses database polling; production requires workload quotas, task retry/dead-letter logic, metrics, and separated AI compute. `create_all` is not a schema migration system. Encryption currently derives from `APP_SECRET`; separate rotatable KMS keys are recommended before production deployment.

# AI Resume Parser — Source-Code Audit, Gaps, and AI/RAG/ML Strategy

**Assessment date:** 15 October 2026  
**Input:** `ai-resume-parser.zip` → `New_Updated_CV/`  
**Scope:** Static review of the application's Python source, configuration, dependencies, README and automated tests; selected executable unit tests and controlled synthetic end-to-end scenarios. No production deployment, live local/cloud LLM, or actual external recruiter was available for assessment.

## 1. Executive assessment

This is a functional, explainable **batch CV matching and shortlisting prototype**, not yet a production Applicant Tracking System (ATS). Its principal strength is a curated, evidence-backed deterministic scoring pipeline with an **optional** LLM pass for uncertain conceptual matches. It should be retained and improved as a baseline, rather than discarded in favor of generative AI.

**Present:** PDF/DOCX/TXT extraction; limited OCR; JD parsing; skill vocabulary/aliases; contextual inference; role-by-role experience estimation; education extraction; hard gates; weighted ranking; evidence snippets; duplicate-by-file-content detection; CSV export; local Ollama and cloud LLM clients; several security safeguards; Pytest cases.

**Not present in the application source:** embeddings, vector search, a RAG retrieval-and-generation pipeline, a trainable ML model, candidate/search database, multi-user/multi-tenant architecture, persistent candidate lifecycle, comprehensive recruitment workflows, a backend HTTP API, a worker queue, versioned model governance, measurable ranking evaluation against recruiter labels, deployment/CI artifacts.

**Release assessment:** useful controlled internal demonstrator; not ready for external multi-customer recruitment screening. Ranking percentages are heuristic suitability indices, not predictive likelihoods or calibrated employment outcomes.

## 2. Source inventory and execution path

The extracted source contains **8,687 lines of Python** across application code, configuration and tests: ~5,966 core module lines, 747 UI lines, 231 config lines and ~1,743 test lines. The ZIP also includes a large Windows `.venv/`, `.git/` history and sample candidate CVs. The 242 MB archive is not a clean source distribution.

| Path | Role | Notes |
|---|---|---|
| `app.py` | Streamlit UI | Inputs, tables, detailed breakdown, PDF rendering, CSV, theme, password gate |
| `app_llm.py` | Auto-LLM Streamlit entrypoint | Can trigger the LLM pass after rule evaluation |
| `config.py` | Thresholds, weightings, model/client limits | Tuning surface; some defaults differ from README prose |
| `modules/extractor.py` | PDF/DOCX/TXT parsing and OCR | Validates type, bytes, PDF pages, DOCX expansion |
| `modules/jd_profile.py` | JD extraction | Requirements, alternative tools, mandatory/nice-to-have, degrees, titles, experience minimum |
| `modules/skill_extractor.py` | Skill taxonomy | Canonical aliases, ambiguous terms and technology names |
| `modules/matcher.py` | Requirement matching | Direct, related, inferred, composite stacks, CV evidence |
| `modules/experience.py` | Employment history | Date ranges, overlapping job months, undated roles |
| `modules/education.py` | Qualifications | Degree labels, fields, years, in-progress status |
| `modules/scoring.py` | Heuristic score | Skills/experience/education/title, must-have caps, category assignment |
| `modules/llm_engine.py` | LLM verifier | Ollama/cloud clients, structured output, numbered-line evidence, caching |
| `modules/pipeline.py` | Orchestrator | JD → files → profiles → ranking → optional LLM refinement |
| `tests/*` | Automated test suite | Parsing, scoring, JD, matcher, LLM, UI and mixed fixture scenarios |

**End-to-end:** Recruiter submits JD + files → JD parsed → per-file validation/extraction (OCR when possible) → duplicate hash check → education and employment extraction → infer JD-relevant experience → deterministic skills/evidence → hard gates → weighted score/caps → sortable candidate list → optional LLM checks *only unmatched concept requirements* for limited candidates → evidence revalidation → rerank/export.

## 3. Verified strengths (preserve these)

1. **Explainable baseline.** For every matched JD requirement the system retains a corresponding CV text snippet and classifies direct/inferred/related evidence. This is more auditable than an unexplained embedding-only score.
2. **Conservative LLM role.** The optional model reviews a shortlist of only *unmatched concepts*. It is not trusted to invent resume experience. It points to numbered CV lines and low-confidence/invalid claims are ignored.
3. **Practical parsers.** PDF reading-order handling, DOCX paragraphs inside tables, normalized Unicode, OCR fallback, file checks and per-file failure isolation are valuable components.
4. **Relevant versus total experience.** Overlapping periods are merged; roles can be judged against the JD; undated roles are highlighted, not silently awarded years.
5. **Multiple requirement forms.** JDs can express alternatives, technology stacks, examples and priority levels. Alias handling and directional inference avoid some naive keyword mismatches.
6. **Security-aware implementation.** Signature/size checks, DOCX expansion protection, XSRF configuration, output escaping, safe CSV formula prefixes and basic prompt-injection checks are present.
7. **Graceful fallback.** An unavailable or invalid LLM normally retains the deterministic result. The system has retries, timeouts, batch budget checks and a circuit breaker.

## 4. Prioritized findings — observed vs recommended

| Severity | Finding / evidence | Consequence | Required response |
|---|---|---|---|
| **P0** | `.streamlit/secrets.toml` **inside uploaded archive contains a populated LLM_API_KEY**; `.git/` also included. | Credential exposure and possible unintended cloud usage. | Rotate/revoke key immediately, audit Git history, exclude secrets, CV fixtures and environments from distributable ZIPs. |
| **P0** | `modules/llm_engine.py:prepare_cv_lines()` strips some contact data but a synthetic test confirmed it **retains gender, nationality and address fields**. | Sensitive personal data may reach cloud model; privacy notice overstates removal. | Consent/approved provider and region, strict field redaction, redaction tests, minimization and no-logging controls; optionally disable cloud by default. |
| **P0** | `modules/pipeline.py:analyse_document()` checks `bachelors_required and not any_degree`; in-progress degrees count as detected, while `graduates_only` is a separate gate. A synthetic in-progress bachelor's CV scored **78.57% Highly Suitable** for a JD saying bachelor's required. | Incorrect eligibility interpretation and consequential shortlisting errors. | Add a recruiter-editable degree completion requirement; separate *completed*, *in-progress*, *unknown*; avoid silent auto rejection. |
| **P1** | `app_llm.py` can auto-run optional AI verification when a cloud key is configured; the cloud notice is not a strong pre-transmission consent gate. | CV text can be transmitted before an explicit per-batch acknowledgment. | Require opt-in and clear provider/data-boundary information before sending CV content. |
| **P1** | Single shared-password protection in `app.py:require_login()`, with `login_attempts` in session; no identities, role permissions or tenant scoping. | Not sufficient for customer-facing recruitment SaaS. | OIDC/SSO, RBAC, server-side rate limit, per-tenant authorization, audit log, TLS and controlled networking. |
| **P1** | No recruiter verification/edit workflow for parsed JD, extracted experience, skill evidence or score overrides. | Parser mistakes silently affect rankings. | Add review/edit queues, evidence location and manual decision/override with reason. |
| **P1** | `Candidate` and requirement matches include quotes but no persistent page, bounding box or normalized source span provenance. | Hard to audit/verify an AI claim against the original document. | Source-span + page/paragraph coordinates and immutable extraction versions. |
| **P1** | Cloud text preparation and `LLM_MAX_CV_CHARS` truncate/filter lines; pipeline has a 100k-character analysis cap. | Relevant later-page material can be missed without a coverage completeness measure. | Coverage reporting, document section segmentation, retrieve relevant chunks across the full document. |
| **P1** | Config weights 45/32/13/10, must-have policy and related-skill credits are hand-set, not calibrated against adjudicated labels. | Overstated significance of a percentage; potential systematic bias. | Introduce labeled evaluation sets, rubric-based scorecards and human review; describe score as decision support only. |
| **P1** | Direct named skills listed without work evidence usually earn full direct credit; role relevance uses work-section heuristics. | Self-declared keywords can be confused with verified competency. | Separate `listed`, `demonstrated`, `verified` and `assessed` evidence levels. |
| **P1** | `st.cache_data` on uploaded JD/preview and module-level LLM cache; theme-restore uses a global resource dictionary with session snapshots. | Personal data retention is not confined only to a session as the README implies. | Document all retention locations; bound TTLs, add purge/logout, tenant separation, avoid caching unredacted PII. |
| **P1** | Regex-based prompt-injection detector has finite patterns, and the LLM's 'high' confidence is self-reported. | Attacks and false-positive AI evidence may slip through. | Treat all uploads as untrusted; layered guardrails, semantic validity checks, grounded citations, human review and adversarial tests. |
| **P1** | Model result cache key uses `(model, CV text, requirements)` but not endpoint/provider, tenant, prompt or rule version. | Stale/cross-configuration verdicts; invalidation and reproducibility problems. | Include tenant, provider, model revision, prompt hash, schema and pipeline version; establish lifecycle/TTL. |
| **P1** | Batch `budget_seconds` checked before future batches, but the threadpool can wait for running calls. | The budget is not a guaranteed request deadline. | Deadline-aware tasks, cancellation, queue processing, request limits and cost/token telemetry. |
| **P2** | Duplicate detection is SHA-256 identity of bytes only. | Same candidate with regenerated PDF or multiple versions appears more than once. | Fuzzy/text duplicate detection + human-approved candidate identity merge. |
| **P2** | Date calculations depend on `date.today()` and present-time work intervals. | Scores drift without recorded evaluation date. | Save evaluation timestamp, extracted fact versions and model/rule versions for repeatability. |
| **P2** | Parser relies on English patterns and software-heavy taxonomy; legacy DOC and rich tables/graphics remain imperfect. | Cross-domain and multilingual recall remains limited. | Language-detection, multilingual parsing, layout-aware extraction, taxonomy packs. |
| **P2** | No Postgres schema, migrations, API, queue, deployment/CI files, user permissions, stored applications, notes or interview workflow. | No durable multi-user recruitment product or scalable back office. | Add product persistence, API, workers, operations and hiring workflow iteratively. |
| **P2** | No end-to-end independent benchmark across protected groups, roles, formats, layouts and locale. | Accuracy/performance and disparate impact cannot be established. | Build consented, anonymized gold data and measured acceptance criteria before production screening. |

### Documentation mismatches

- README suggests minimum experience is full credit, but `modules/scoring.py:experience_component()` uses `EXP_AT_MINIMUM=0.85`, rising toward full credit over additional years.
- README simplifies the must-have cap; the code uses a continuous soft cap with multiple thresholds (35%, 65% coverage and dynamic ceilings).
- README suggests CVs live only in each user's session; application data can also be in server-side Streamlit caches, a module-level LLM cache and theme-restore snapshots.
- README uses 'semantic' to describe hand-curated inference. This is valid in ordinary language, but it is **not semantic embedding retrieval**.

## 5. What AI, RAG, embeddings and ML actually mean here

| Approach | Implemented? | Correct role in this product |
|---|---|---|
| OCR + document layout | **Yes (basic)** | Extract text; upgrade to page-aware layout and reliability flags. |
| Rule-based/semantic-taxonomy skill detection | **Yes** | High-precision exact skill evidence, alternatives, hard constraints. |
| LLM reasoning / extraction | **Partially** | Current verification of unmatched concepts. Extend to structured extraction and JD clarification *under a verification boundary*. |
| Embeddings | **No** | Search similar job titles, projects and experience descriptions; retrieve candidates beyond keyword overlap. |
| Vector database | **No** | Persistent similarity index for consented CV/JD chunks. |
| RAG | **No** | Retrieve indexed CV/JD/company policy evidence and synthesize a cited answer without inventing facts. |
| Reranking / pretrained cross-encoder | **No** | Re-evaluate retrieved candidate-to-JD evidence for closer textual relevance. |
| Supervised learning-to-rank | **No** | Optional later stage when legitimate, representative and validated human rubric labels exist. |
| Prediction of hire success / personality | **No** | Not recommended without a validated and legally permissible use case; avoid sensitive/proxy inferences. |

**Important distinction:** `matcher.py` recognizes RAG as a skill to match in *someone's CV*. That does not make this application a RAG system.

## 6. Recommended AI feature backlog

### A. Structured CV profile extraction (first AI feature)

Keep deterministic extractor as source and have an LLM produce a **strict JSON proposal** (never an unquestioned profile) including:

- Work entries: employer, role, normalized start/end, currently employed, work description, provenance.
- Qualification entries: title, field, institution, completion date, **completed/in-progress/unknown**, provenance.
- Skills: normalized canonical ID; direct mentions, sections, projects, years claimed, evidence strength and source spans.
- Certificates/projects: dates, links, supporting evidence, parser confidence and uncertainty reasons.

Validate against the source PDF/DOCX spans. Mark ambiguous data as **needs review**. Do not assume an absent fact means the candidate lacks it. Keep personal contact information in a separate access-controlled record rather than mixing it into the LLM matching prompt.

### B. AI JD interpretation assistant

Extract requirements into reviewable structured records: mandatory, preferred, alternatives, minimum vs desired experience, educational completion, geography/availability, work authorization, and business-specific criteria. Detect ambiguous phrases and contradictions, then ask **recruiter** to confirm; do not let the model silently invent filters.

### C. Evidence-oriented comparison

Produce a matrix: `requirement | status: evidenced / potentially evidenced / missing from document / needs clarification | source quote | page | human decision`. Do not equate 'not mentioned' with 'not possessed'. Keep the score alongside its versioned rubric, and provide a manual review pathway for any hard-gate failure.

### D. Candidate discovery and natural-language search

Allow requests such as "Find applicants with hands-on Python API work and approximately three years of relevant experience, excluding unverified skill-list-only mentions." Use structured filters plus hybrid search rather than simply asking an LLM to pick candidates from a concatenated list.

### E. Recruiter copilot with RAG

Ground conversational answers in **authorized** candidate document chunks, their source coordinates, job requirements and configured HR policies. Return citations to the uploaded document, distinguish evidence/interpretation, log the source IDs and abstain when evidence is insufficient. Never send one tenant's documents into another's retrieval context.

### F. Supporting workflows

Generate interview topics from missing evidence, standardized non-discriminatory interview guides, candidate-summary drafts, duplicate warnings and skill-gap questions; leave shortlist decisions, outreach and rejection communications under human control.

## 7. Proposed RAG architecture

**Do not use RAG for the simple task of comparing one short JD with one CV**; structured extraction + evidence matching is more controllable. RAG is valuable as the CV collection, policies and queries grow.

Recommended hybrid search path:

```text
Original file → validated parser/OCR → sections + source/page positions
                                   ↓
                            Normalized facts
                                   ↓
                     Tenant-scoped Postgres records
                                   ↓
                       Section-level CV text chunks
                                   ↓
               Embeddings → PostgreSQL + pgvector index
                                   ↓
Recruiter query → authZ + filters → keyword/full-text + vector retrieval
                                   ↓
                 merge/rank (RRF) → reranker (optional)
                                   ↓
          evidence verifier + deterministic requirement rubric
                                   ↓
             grounded candidate results / optional LLM answer
                                   ↓
                     human reviewer + audit trail
```

- **Chunk by semantic section** (role, project, degree) while retaining candidate ID, document version, section, offsets/page and tenant, not arbitrary 1,000-character slices.
- **Separate structured truth from narrative search.** Age, degree completion and experience minimums should not be established by cosine similarity.
- **Hybrid retrieval**: Postgres full-text/BM25-equivalent lexical approaches + pgvector HNSW search, with optional reciprocal-rank fusion/cross-encoder reranking. The pgvector docs explicitly support Postgres full-text hybrid retrieval.
- **Explicit cross-tenant authorization** before vector retrieval and LLM context assembly. Partition/index carefully at scale; pgvector notes multitenancy affects recall and speed.
- **Deletion and versioning:** delete source files, chunks and embeddings together on authorized erasure; re-embed when model changes; rebuild when CV changes.
- **Avoid embedding PII** if it is not necessary for matching; keep identity and contact fields separated.

**Initial infrastructure choice:** keep the existing Python parsing/scoring service; add Postgres+pgvector, S3-compatible private object storage for CVs, an API plus worker queue. A Streamlit UI is fine for a controlled proof-of-concept. For multi-customer SaaS, a separate authenticated React/Next.js interface with a NestJS or FastAPI gateway is more flexible. Do not rewrite mature Python extraction rules into JavaScript simply to align the UI stack.

### Suggested minimal persisted tables

`tenants`, `users`, `roles`, `jobs`, `job_requirement_versions`, `candidates`, `applications`, `documents`, `document_sections`, `employment_entries`, `education_entries`, `skill_evidence`, `embeddings`, `match_runs`, `match_requirements`, `human_reviews`, `review_overrides`, `consents`, `audit_events`, `processing_jobs`.

**Important fields:** every evaluation should preserve `tenant_id`, `job_version`, `cv_version`, `ruleset_version`, `model_id`, `prompt_version`, `retrieval_version`, `evaluated_at` and exact evidence IDs. Put tenant scope in both SQL policies and API authorization.

### Suggested API contracts

- `POST /jobs`, `POST /jobs/{id}/interpret`, `PATCH /jobs/{id}/requirements` (human confirmation)
- `POST /candidates/import`, `GET /candidates/{id}`, `DELETE /candidates/{id}` (subject to retention policy)
- `POST /jobs/{id}/match-runs`, `GET /match-runs/{id}/results`
- `POST /candidates/search` (tenant-scoped structured + semantic filters)
- `POST /match-runs/{id}/overrides` (authorized human review)
- `POST /assistant/query` (retrieval-grounded answer with citations)
- `GET /audit-events` (restricted)

## 8. Recommended ML strategy

**Stage 0 — objective baseline:** preserve rules and freeze the scoring rubric for benchmarking.

**Stage 1 — pretrained models without fine-tuning:** use a locally hosted or privacy-approved multilingual embedding model for candidate and project text; evaluate candidate recall. Add a pretrained reranker only if retrieval results warrant its latency/cost. No custom training is required at this stage.

**Stage 2 — calibration / learning-to-rank only with sufficient data:** collect independent, consented recruiter **rubric labels** on representative CV-JD pairs. Avoid training directly on historic "hired/not hired" outcomes because of selection bias, confounding and possible historical discrimination. Start with interpretable calibration or ranking and compare against the rules baseline.

**Stage 3 — monitored production quality:** evaluate drift by JD family, document layout, language, skill taxonomy revision, model update and recruiter disagreement. Protected-characteristic fairness analysis, if legally allowed and appropriately governed, should use segregated audit workflows—not model input features. Ensure humans can appeal/override results.

**Not recommended now:** training a resume-ranking classifier on the sample resumes included in the ZIP, creating synthetic 'ground truth' labels with an LLM, predicting demographics/personality from CVs, or using one embedding cosine score as the final hiring decision.

## 9. Evaluation and test report

### What I executed

- Static inventory of the source modules, README, requirements and application security configuration.
- Python syntax compilation: passed for the extracted source and tests.
- Selected Pytest modules with ordinary local dependencies available: `test_scoring.py` **19 passed**, `test_jd_profile.py` + `test_llm_engine.py` **48 passed**, `test_experience_education.py` **35 passed**, `test_hr_judgement.py` **13 passed**, `test_non_tech_jds.py` **9 passed**, `test_skills_and_matching.py` **108 passed**. **At least 232 tests passed** across these completed subsets; this does not imply the full suite passed.
- Other tests started successfully, but broad suite runs were time-limited while repeatedly evaluating sample PDF batches. I did not establish a full-suite passing result.
- Streamlit and RapidOCR were unavailable in the inspection runtime; interactive Streamlit UI, actual OCR and real LLM round-trip were **not fully executed**. The ZIP's Windows virtualenv is not a substitute for a clean isolated deployment test.
- A synthetic Python+PostgreSQL+Django CV against a related JD scored ~97%; unrelated sales CV scored 0; the feature demonstrated coherent baseline behavior on this controlled example.
- Another synthetic candidate whose bachelor's was explicitly *in progress* scored **78.57% Highly Suitable** despite a JD specifying a bachelor's requirement.
- Synthetic cloud redaction check showed **gender, nationality and address** remained in text sent for LLM verification.

### Missing quantitative evaluation

Create a labeled test set spanning technical and non-technical JDs, multiple resume layouts, image-only PDFs, multilingual cases, unusual employment histories, alternative technology requirements, unstated facts, older/newer CV versions and adversarial documents. Define approval criteria before deployment:

| Area | Suggested measures |
|---|---|
| Parsing | degree completion accuracy, job date accuracy, relevant-years absolute error, OCR failure rate |
| Retrieval | recall@k, precision@k, mean reciprocal rank, normalized DCG |
| Evidence | citation correctness, hallucinated-claim rate, evidence coverage, source-grounding precision |
| Ranking | agreement with independent recruiter rubric, false-negative/false-rejection rate, calibration |
| Operations | p50/p95 upload processing time, queue delay, LLM costs, cache hit rate, job failures |
| Governance | override rate, unknown-status frequency, group-level error comparisons where lawful, deletion SLA |

Use strict holdout data. Do not set arbitrary pass-rate metrics or claim reliability based on the current small supplied CV examples.

## 10. Security, fairness and compliance considerations

1. Rotate the populated cloud API key packaged in the ZIP and check source-control history.
2. Make cloud AI opt-in per organization and per run. Explicitly display provider, data category, retention implications and processing location. Use locally hosted models when necessary.
3. Strip direct identifiers and protected/sensitive demographic details before matching or embedding wherever possible; separate identity fields. Redaction must be tested, not assumed.
4. Use organization accounts, MFA/SSO if deployed, RBAC, access logs, rate limits, encrypted object storage and a deletion/retention policy.
5. Do not automatically reject based only on unverified parser output. Expose uncertainty and provide recruiter confirmation for educational qualifications, required experience and other eligibility gates.
6. Keep versioned rubric, provenance and reviewer decisions. Provide candidate notice/rights workflows as relevant to target jurisdiction.
7. If targeting EU recruitment use cases, **AI-powered automated candidate scoring/ranking can be classified high-risk** under the EU AI Act, even when recruiters retain ultimate discretion. Obtain jurisdiction-specific legal review; do not assume that calling the system 'decision support' removes compliance obligations. Source: official EU AI Act Service Desk, Annex III and the employment classification guidance.

References:  
- EU AI Act employment examples: https://ai-act-service-desk.ec.europa.eu/en/employment-0  
- EU AI Act Annex III: https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3  
- pgvector documentation: https://github.com/pgvector/pgvector  
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework

## 11. Recommended implementation order

| Phase | Priority | Scope | Completion criterion |
|---|---|---|---|
| **0. Secure** | P0 | Credential rotation; safe source packaging; cloud opt-in; redaction; strict degree-completion logic | No secrets in artifacts; privacy test suite passes; in-progress degrees no longer silently satisfy completed-degree gates |
| **1. Trustworthy parser** | P0/P1 | Structured evidence spans, degree/experience review, rule versioning, recruiter correction | Every high-impact score/gate is traceable and correctable |
| **2. Evaluation harness** | P1 | Gold dataset, comparison dashboard, regression/performance/adversarial tests | Measured baseline and documented release thresholds |
| **3. Product foundation** | P1 | Auth/RBAC, tenant-aware API, DB, file storage, queue, audit/deletion | Persistent, permissioned candidate and job workflows |
| **4. Embeddings/search** | P2 | Section-aware chunks, pgvector, tenant-scoped lexical+vector retrieval | Measured improved candidate discovery recall |
| **5. Grounded RAG copilot** | P2 | Retriever, reranker, source evidence, grounded answers, refusal/abstention | Answers cite authorized original evidence and can be audited |
| **6. Advanced ML** | P3 | Optional calibrated rerank, supervised learn-to-rank if dataset and legal case justify | Outperforms vetted baseline without materially worsening false exclusions/fairness |

**Preserve:** extraction modules, JD logic, skill taxonomy, evidence-aware matching and conservative LLM verifiers. Refactor into stable typed domain objects and testable services, rather than rewriting the entire project.

## 12. Immediate developer task list

1. Remove and rotate the exposed API credential and sanitize package history.
2. Add unit tests for JD requiring completed bachelor's vs in-progress bachelor's, master's, unknown year, equivalent qualifications and optional-degree wording.
3. Make the result tri-state: `met`, `not evidenced`, `needs review`; replace hard automatic exclusion for uncertain parse results.
4. Introduce separate cloud consent and protected-attribute suppression tests (gender, nationality, exact address, DOB, contact, photos, URLs, identifiers).
5. Add `source_id`, `page`, `section`, `start/end offset` to extracted facts and requirement matches.
6. Add a JD interpretation override editor and candidate fact correction UI.
7. Freeze ruleset/model version and evaluation timestamp in output and CSV.
8. Add data retention cleanup, tenant-scoped cache keys and documented privacy boundaries.
9. Improve full test execution times using cached parsed fixtures and separated fast/slow/integration test groups.
10. Once privacy/reliability gates pass, introduce PostgreSQL persistence, hybrid retrieval and only then a RAG chat interface.

## Final position

The largest improvement is **not adding RAG to everything**. The highest-value work is making the existing pipeline trustworthy, measurable, editable and secure. After that, **embeddings and hybrid retrieval** will meaningfully improve discovery across large CV collections, and **RAG** can provide source-grounded recruiter answers. Custom **ML ranking** should come last, and only after collecting valid labels, validating fairness and establishing an appropriate governance process.

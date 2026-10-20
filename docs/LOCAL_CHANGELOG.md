# Implementation Record — Direct-Launch Local Edition

## Major changes vs Enhanced v2

1. Replaced `platform_ui.py` authenticated API client with the new single-process **direct-access Streamlit recruitment studio**; no login, tenant identifier, JWT, bootstrap or backend service.
2. Added Inovera-inspired branded sidebar, hero, cards, configurable local dashboards, full-page candidate profiles and charts.
3. Added persistent local SQLite job, candidate and analysis management in `studio_store.py`.
4. Added multi-file and ZIP import, bounded expansion, safe filenames, duplicate detection, extraction validation and upload progress.
5. Added candidate management and PDF page-at-a-time preview (safe rasterized image rather than executable embedded document).
6. Added direct use of existing `evaluate_batch` including progressive status and optional local Ollama verification.
7. Added typed, JSON-serializable report schemas with separate confidence-weighted skill percent, per-skill JD match percent, relevant experience scoring, total-experience comparison, document evidence, flags and rejections.
8. Added filterable ranking tables with Streamlit percent progress columns and individual result cards with full-width score bars.
9. Added full profile: scores, skill evidence, work history, education, CV viewer, verification and reviewer suggestions.
10. Added statistics for distribution, suitability categories, top missing skills, recruiter recommendations, CSV/JSON downloads and a full JSON viewer.
11. Fixed original category-symbol mismatch between engine statuses and aggregated visualization counters.
12. Added candidate erasure propagation to saved historical result snapshots and job deletion cascades.
13. Added fictional demonstration CVs and a one-click local demo workspace loader.
14. Added Windows/Linux startup scripts, installation requirements, local deployment docs and regression tests.

## Deliberately retained

`modules/` deterministic logic; original `app.py`; optional `api/` authenticated backend source and docs (unused by local edition). Do not launch `uvicorn` for the Studio. The new direct-launch UI does not call any API. Existing cloud credential support in the legacy app/API is not used by this UI.

## Deliberate limitations / future backlog

- **Not a multi-user or hosted SaaS**. No login means it must be localhost-only. No filesystem encryption is provided by the new SQLite store.
- Original matching semantics and heuristic thresholds are preserved; this is not a new statistically trained ML ranker or hiring prediction model.
- No generative RAG chat in local UI; the previous authenticated API source includes a separate retrieval proof-of-concept but is not launched here.
- No browser automation screenshot validation in this release environment; install Streamlit on target machine and complete UI acceptance checklist.
- Scanned documents depend on separately installed OCR; GPU acceleration is not configured.
- No ATS email outreach, hiring decisions, interview scheduling, or external HR integrations.
- Candidate corrections/reviews are presented as evidence and recommendations; there is no persisted human decision workflow in the new single-user UI.

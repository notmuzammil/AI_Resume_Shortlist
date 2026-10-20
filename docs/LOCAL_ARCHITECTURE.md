# Local Studio Architecture

```
Browser (127.0.0.1:8501)
         │
         ▼
Streamlit platform_ui.py (single user, no login)
         ├── studio_store.py (SQLite, jobs / candidates / runs)
         ├── studio_core.py (ZIP upload, normalization, reports, charts data)
         └── modules/pipeline.py
                ├── jd_profile.py
                ├── extractor.py → PDF / DOCX / TXT (optional OCR)
                ├── education.py / experience.py
                ├── skill_extractor.py / matcher.py
                ├── scoring.py
                └── optional local Ollama verification
```

No web API is required or called by `platform_ui.py`.

## Storage schema

`jobs(id, title, description, created_at, updated_at)` — job definitions; updating a JD does not retroactively modify historic reports.

`candidates(id, filename, sha256 UNIQUE, filetype, content BLOB, created_at)` — SHA-256 deduplicated raw files, local to one workspace. On import all files are passed through the original extraction guards. The BLOBs are not encrypted by the application.

`runs(id, job_id FOREIGN KEY, created_at, report_json)` — historical snapshot of profile, normalized candidate score/evidence details, statistics and review suggestions. Foreign key deletion cascades from jobs to runs. Deleting a candidate removes their result from every report and recalculates its aggregate statistics.

The SQLite file is `.studio_data/workspace.sqlite3` by default, or `<RESUME_STUDIO_DATA_DIR>/workspace.sqlite3` if configured. Protect it like the confidential raw CV collection it is. No migration framework is included; use a fresh dedicated database in this release.

## Scoring pipeline

For each run, `evaluate_batch` parses the JD once and then extracts CVs concurrently, identifies duplicate file bytes, scores skills/experience/education/title, applies eligibility and must-have caps, and ranks deterministically. The UI does not recalculate or change the final numeric score. It adds comparisons, evidence presentation, analytics and JSON serialization.

## Security design and limitations

The only network listener is Streamlit on `127.0.0.1`. Multi-user and public deployment without auth are unsafe. ZIP entries are never written to disk by path; path traversal entries are rejected and extraction is bounded by per-file and cumulative size. HTML interpolated with custom CSS is escaped. CV previews use server-side rasterization and text widgets instead of embedding raw PDFs into browser iframes. CSV formula prefixes are escaped for commonly dangerous leading characters.

The operator must manage access, lawful processing, retention, backups and audit obligations. This local edition has no application-level encryption, access logging or server-side per-user permissions. It is not suitable for public recruitment operation without further controls.

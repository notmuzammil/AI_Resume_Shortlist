# Direct-access Recruitment Studio — Features, Screens and Use Cases

## Screens

### 1. Overview
The first screen has an Inovera Labs-styled hero (teal gradient with amber accent in the sidebar), counters for job descriptions, candidate CVs and reports, quick-start cards and a recent-match summary. There is no administrator onboarding and no account creation. A one-click synthetic demo loader fills the workspace with fictional candidates for feature exploration.

### 2. Jobs
Create a job using a pasted JD or an attached DOCX/PDF/TXT JD. The JD parser identifies role title, required and inferred experience, education fields and skill requirements with importance (must/normal/nice). Displays suggestions when requirements are underspecified. Users can inspect, edit, evaluate or delete a job. Prior report snapshots remain historical after a job edit.

### 3. CV Library
Multi-select original resume files and ZIPs. Validates file types, signatures, uncompressed size, document limits and readable text. Deduplicates byte-identical files by SHA-256, stores originals locally, lists and filters imports, previews PDF pages or extracted text, downloads originals, and deletes CVs plus any saved reports containing that candidate.

### 4. Match Lab
Select one stored job and any/all saved candidates (200 per matching run by default), and start matching. A progress bar advances after each CV completes. The existing weighted deterministic evaluator applies JD interpretation, skills, education and professional experience, then status and score bands. The UI lists each candidate with a horizontal percentage bar, numeric progress columns, filterable score threshold, categories and a detail button. Errors and duplicate/rejected statuses are visible separately. It saves all runs locally and exports ranked CSV and detailed JSON.

Optional AI double-check: Enable local Ollama and specify an installed model. This only attempts to verify otherwise-unmatched *conceptual* skills; it does not guarantee matches or replace the deterministic evaluator. Model not installed/service offline: results fall back to the rules engine, with warnings.

### 5. Candidate Profile
Select a run and candidate. Five tabs:
1. **Match overview:** final score, number of skills satisfied, relevant and total years, progress bars for scoring components, experience comparisons and cap explanations.
2. **Skills & evidence:** a separate 0–100% bar for each requirement; mandatory/preferred tags; direct/inferred/related status; evidence quotes, basis and hands-on verification cues.
3. **Experience & education:** extracted degree names, dated roles, months, role relevance and undated-role exclusions.
4. **Original CV:** PDF page selector with server-rasterized preview, or read-only DOCX/TXT extracted text. Original file download.
5. **Review suggestions:** missing mandatory requirements, unproven hands-on experience and date/qualification verification notes. Automated scoring never advances hiring stages.

### 6. Reports & Insights
Report chooser, candidate population statistics, histogram, suitability category counts, most commonly unmatched JD skills, recommendations, detailed JSON viewer and CSV/JSON downloads. Statistics only count successfully scored candidates for score averages/missing skills.

## Key use cases

**UC-01 — Recruiter starts immediately**: Open Streamlit on localhost, see dashboard without login. No API process required.

**UC-02 — Vacancy creation**: Save title and JD; inspect parsed must-have/optional requirements; revise job scope.

**UC-03 — Bulk candidate import**: Select 50 PDF resumes or a ZIP; reject invalid and duplicate files, store valid originals.

**UC-04 — Repeat evaluation**: Match the same pool against different roles without uploading files again.

**UC-05 — Compare relevant and total experience**: Display both year counts, JD reference, and independent normalized bars. Total-year bar is expressly informational and never substituted for JD-relevant years.

**UC-06 — Evidence review**: Examine a candidate's requirement-by-requirement confidence/evidence; check PDF pages; determine which skills need interview verification.

**UC-07 — Rank/export**: Sort candidate scores, filter for thresholds, export Excel-compatible sanitized CSV and JSON with numeric score components.

**UC-08 — Quality review**: Detect major market-wide gaps (frequent missing skills) and review potentially over-constrained job requirements.

**UC-09 — Privacy cleanup**: Delete a candidate and automatically remove them from existing saved report JSON; perform wider backups/export disposal separately.

**UC-10 — Resume OCR**: If optional OCR runtime is installed, process supported scanned PDF pages. Otherwise receive a readable processing issue.

**UC-11 — Optional local concept recheck**: Use a locally running Ollama model; never send candidate text to a cloud LLM from this UI.

**UC-12 — Reopen saved results**: After restarting Streamlit, previously created jobs and reports appear from the local SQLite workspace.

## Non-goals for this release

No login/auth/roles, organizations, SaaS tenants, public site, HRMS integration, auto-emailing, interview scheduling, predictive hiring, trained ML candidate ranking, or generative RAG chat. Existing archived API source may contain some enterprise prototypes, but the Studio UI does not claim to implement or launch them.

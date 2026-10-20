# Studio Testing and Acceptance

## Commands

`python -m pytest -q tests/test_studio.py` exercises new local features using an isolated temporary SQLite workspace. `python -m pytest -q tests --ignore=tests/test_app.py` runs original parser/matcher regressions as well; tests requiring unavailable user-specific CV fixtures or optional integrations are skipped.

## New automated tests

- Create, update, delete and validate job descriptions.
- Import documents, SHA-256 duplicate suppression, original-file data integrity.
- End-to-end JD/CV pipeline → weighted candidate details → stored JSON and CSV exports.
- Category counter correctness, percent bounds and distinction between total/relevant experience.
- Deleting candidates purges their JSON records from historical reports; deleting a job cascades match analyses.
- ZIP path traversal protection, invalid ZIP handling, supported file filtering and data verification.
- CSV formula injection protection, normalized filenames and absence of authentication tables in the local database.
- JD requirements and output types (including source evidence and individual percentage scores).

## Known verification boundary

The build environment has Python and pytest but **does not contain Streamlit**. Installing Streamlit was attempted; this environment cannot access the package repository. All new local model/store workflows and legacy tests runnable without the Streamlit dependency were exercised. The Streamlit UI code was syntax-compiled, but browser-rendered navigation, progress bars, responsive layout, PDF viewer widgets and Windows startup cannot be honestly reported as executed. Those require local validation on the target machine.

## Manual UI acceptance checklist (run on Windows)

- [ ] Double-click `start_windows.bat`: dashboard opens without login.
- [ ] Save two jobs and check their parsed skill/experience requirements.
- [ ] Import multiple PDFs/DOCX/TXT and a ZIP; verify progress, duplicates and malformed-file errors.
- [ ] Run matching; check score and rank ordering with progress bars, filters, and multiple job descriptions.
- [ ] Open each candidate's profile; inspect each skill percent and evidence, relevant vs total experience, education and rejection flags.
- [ ] Preview a multi-page PDF, switch pages and download the original.
- [ ] Inspect report dashboard charts and useful recruitment suggestions.
- [ ] Export CSV/JSON and verify numerical percentages, evidence fields and rejected-document records.
- [ ] Restart the application; ensure jobs, CVs and reports remain.
- [ ] Delete a candidate and inspect a previous JSON report to ensure their entry is removed.
- [ ] Confirm app listens on localhost only and `.studio_data` is not included in backups intended for public sharing.

## Safety checks

No default admin credentials are supplied because the user explicitly requested no authentication. Protect the SQLite file and the Windows/macOS/Linux account. Automated ranking is not an autonomous hiring decision.

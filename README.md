# Inovera Labs — Talent Intelligence Studio (Local Edition)

**Direct-launch, single-user recruitment dashboard. No admin account, login, JWT, API server, external database, or cloud AI account is needed.**

This release extends the existing AI Resume Parser with a redesigned Streamlit application for creating jobs, bulk-importing resumes, analyzing JD/CV fit, reviewing numeric scores and evidence, previewing CVs, and exporting visual ranking reports. The original parser, OCR handlers, skill matching, experience calculations and score caps are retained.

## Start on Windows (recommended)

1. Extract the entire ZIP into a regular directory, e.g. `D:\Projects\TalentIntelligence`.
2. Install **Python 3.11 or 3.12**, ensuring `py` or `python` works from a command prompt.
3. Double-click **`start_windows.bat`**. On first run it creates `.venv` and installs the Studio dependencies (internet required for package installation).
4. Open **http://127.0.0.1:8501** if the browser does not open automatically. There is **no login screen**.
5. Your jobs, CVs and match reports persist in `.studio_data/workspace.sqlite3`.

### Manual installation (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-studio.txt
streamlit run platform_ui.py --server.address=127.0.0.1
```

If `py -3.11` is unavailable but Python 3.12 is installed, use `py -3.12` or `python`.

### Linux / macOS

```bash
bash start_linux.sh
# Or manually: python3 -m venv .venv; .venv/bin/pip install -r requirements-studio.txt
# .venv/bin/streamlit run platform_ui.py
```

With an empty workspace, the **Shortlist** page offers **Try with sample data**: a demo job and four fictional CVs.

## What to click

The app has three pages, in the top bar.

1. **Job**: add the one job description (title plus pasted or uploaded JD), edit it, or delete it to start a different job.
2. **Resumes**: drop PDF/DOCX/TXT files or a ZIP, up to 20 resumes. Invalid, duplicate or unreadable files are reported. "Start fresh" at the bottom deletes all resumes, or everything.
3. **Shortlist**: choose **AI (NVIDIA)** or **Quick, no AI**, tick the consent box for AI, and press **Shortlist**. Each candidate gets a card with score, category, a short summary and the missing must-haves. **Open** shows strengths, concerns, the requirements checked, interview questions, work history and the resume itself. Results download as CSV or JSON.

Light and dark mode follow the computer's setting; switch in the ⋮ menu at the top right.

The scoring percentages are *JD-fit heuristics*, not hiring-success probabilities. Relevant years affect the experience component; the **total experience percentage is a clearly labeled informational comparison**. Candidate names are derived from filenames solely as display labels. Interview and hiring decisions require human review.

## Local data & privacy

- No authentication or authorization is performed. The application binds to **127.0.0.1 only** in `.streamlit/config.toml`. Do not expose the server to the internet or LAN or share it with untrusted operating-system accounts.
- The directory `.studio_data/` contains **unencrypted** raw candidate CV bytes and matching evidence inside SQLite. Restrict filesystem access, back up/erase it appropriately, and obtain an appropriate lawful basis and notice before uploading CVs. Local encryption at rest is **not** provided by this edition; use device/full-disk encryption if required.
- Remove a CV from **Resumes → Open a resume → Delete** to delete it and strip that candidate's records from saved reports. SQLite may retain removed bytes in free pages until vacuum; for high-assurance erasure, delete the database file securely or run an approved data-erasure process. Downloaded CSV/JSON exports are outside the program's deletion control.
- No cloud LLM is called unless you pick **AI (NVIDIA)** in Shortlist and tick the consent box. Then the AI first turns the job description into a rubric (each requirement must quote the JD line it comes from; it is reused while the job text is unchanged), and each readable CV (name line, emails, phone numbers, links, addresses and personal details removed) is sent with that rubric as numbered lines (`modules/ai_evaluator.py`). The AI only gives a verdict and an evidence line per requirement; years of experience, the score, the caps and the ranking are calculated in code, and borderline results are read a second time. CVs whose AI evaluation fails are listed as "not evaluated" instead of being ranked on the rules score, and CVs the rules rejected on eligibility are re-checked by the AI. CVs that contain instructions aimed at AI screeners are never sent. Put the key in `.streamlit/secrets.toml` as `LLM_API_KEY = "nvapi-…"` (or set `NVIDIA_API_KEY`); that file is excluded from Git. Scanned-PDF OCR is optional (`pip install -r requirements-ocr.txt`).
- Limit: 200 CVs per run by default; 10 MiB per CV by default; 150 MiB total uncompressed upload batch; PDFs scanned without OCR can be rejected with a readable diagnostic.
- Local `.studio_data` is automatically excluded from Git and the release ZIP. The archive intentionally excludes `.streamlit/secrets.toml`, historical credentials, real candidate documents, `.git`, and `.venv`.

## Deliverables / architecture

| File | Purpose |
|---|---|
| `platform_ui.py` | Primary, redesigned, direct-access recruitment dashboard |
| `studio_core.py` | Bulk import safety, score JSON projection, statistics and CSV/JSON exports |
| `studio_store.py` | Job/CV/report SQLite persistence, de-duplication and deletion |
| `modules/` | Original deterministic parser, extractor, JD profiler, experience, education and scorer |
| `start_windows.bat` / `start_linux.sh` | Simplified local launchers |
| `requirements-studio.txt` | Lean installation for the new direct-launch UI |
| `docs/LOCAL_*` | Architecture, UI, features, use cases, scoring, testing |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests/test_studio.py
# Optional full regression suite (original fixtures/Streamlit/extra services may be skipped):
python -m pytest -q tests
```

See [`docs/LOCAL_TESTING.md`](docs/LOCAL_TESTING.md). Automated workflow tests are not a substitute for live browser testing; the build environment for this release did not include Streamlit and could not install it, so live visual browser checks remain to be performed on the target machine.

## Read more

- [Local UI, functionality and use cases](docs/LOCAL_FEATURES_AND_USE_CASES.md)
- [Architecture and persistent data](docs/LOCAL_ARCHITECTURE.md)
- [Detailed scoring, skill bars and reporting schema](docs/LOCAL_SCORING_AND_JSON.md)
- [Tests, limitations and acceptance checklist](docs/LOCAL_TESTING.md)
- [Migration, fixes and implementation record](docs/LOCAL_CHANGELOG.md)
- [Original technical audit](docs/ORIGINAL_AUDIT.md)

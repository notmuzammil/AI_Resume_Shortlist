# Verification Report — Talent Intelligence Studio Local Edition

## Automated tests

**2026-10-15**, isolated build environment (Python 3.13 / pytest 9):

```text
python -m pytest -q tests --ignore=tests/test_app.py
268 passed, 43 skipped (23.84 seconds)
```

The **19 new Studio tests** cover jobs, multi-file/ZIP ingestion and safety, CV validation, normalized skill/experience scores, SQLite persistence, runs, candidate erasure, CSV formula escaping, JSON serialization, real synthetic PDF page rasterization, DOCX extraction, a full JD/CV matching workflow and ranked fictional demo data.

The 43 skipped legacy tests depend on original confidential resume fixtures not included in the sanitized source distribution, plus some optional integration prerequisites. An additional legacy UI test file (`tests/test_app.py`) was excluded from this exact command because the build environment has no Streamlit installation.

## Live interface testing limitation

The Streamlit UI source was compiled, but a live browser UI launch could **not** be tested: Streamlit is not installed in this sandbox, and a package-install attempt failed because it has no package-index network access. It must be visually tested on a local Windows/macOS/Linux machine after installing `requirements-studio.txt`. Do not interpret the automated test count as browser verification.

## Packaging and safety

- The release ZIP excludes `.venv`, `.git`, `.streamlit/secrets.toml`, `.env`, `.studio_data`, raw candidate data, and local private storage.
- `platform_ui.py` runs with no login, account/bootstrap or separate API process.
- The primary Streamlit config binds only to `127.0.0.1`.
- `api/` is retained strictly as unused v2 reference code, not part of the direct-launch workflow.
- The project is software for decision support, not an autonomous hiring decision system.

## Reproduce

```bash
python -m pip install -r requirements-studio.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_studio.py
python -m pytest -q tests --ignore=tests/test_app.py
streamlit run platform_ui.py
```

See [`docs/LOCAL_TESTING.md`](docs/LOCAL_TESTING.md) for the checklist.

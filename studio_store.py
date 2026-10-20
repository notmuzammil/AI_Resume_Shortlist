"""Local SQLite persistence for a single-user, localhost-only Streamlit workspace.

No logins, tokens, organization IDs, or external API processes. SQLite stores
original CVs as BLOBs and reports as JSON in the local workspace. The operator
is responsible for protecting this private directory and deleting CVs on time.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / '.studio_data'


def workdir() -> Path:
    path = Path(os.environ.get('RESUME_STUDIO_DATA_DIR', str(DEFAULT_DIR))).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt':
        try: path.chmod(0o700)
        except OSError: pass
    return path


def db_file():
    return workdir() / 'workspace.sqlite3'


@contextmanager
def connect():
    db = sqlite3.connect(str(db_file()), timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys = ON')
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    with connect() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
                filetype TEXT NOT NULL, content BLOB NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL, report_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runs_job_time ON runs(job_id, created_at DESC);
        ''')


def now():
    return datetime.now(timezone.utc).isoformat()


def create_job(title, description):
    from modules.pipeline import prepare_jd
    prepare_jd(description)
    if not str(title).strip(): raise ValueError('Enter a job title.')
    key, stamp = str(uuid.uuid4()), now()
    with connect() as db:
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?)', (key, title.strip()[:180], description.strip(), stamp, stamp))
    return key


def update_job(job_id, title, description):
    from modules.pipeline import prepare_jd
    prepare_jd(description)
    if not str(title).strip(): raise ValueError('Enter a job title.')
    with connect() as db:
        row=db.execute('UPDATE jobs SET title=?, description=?, updated_at=? WHERE id=?',
                       (title.strip()[:180], description.strip(), now(), job_id))
        if row.rowcount != 1: raise ValueError('Job not found')


def list_jobs():
    with connect() as db:
        return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY updated_at DESC')]


def get_job(job_id):
    with connect() as db:
        row=db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        return dict(row) if row else None


def delete_job(job_id):
    with connect() as db:
        db.execute('DELETE FROM jobs WHERE id=?', (job_id,))


def insert_candidate(filename, content):
    from studio_core import clean_name
    digest = hashlib.sha256(content).hexdigest()
    with connect() as db:
        found=db.execute('SELECT id,filename FROM candidates WHERE sha256=?', (digest,)).fetchone()
        if found: return dict(found), False
        identifier=str(uuid.uuid4())
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                   (identifier, clean_name(filename), digest, clean_name(filename).rsplit('.',1)[-1].lower(), content, now()))
        return {'id':identifier, 'filename':clean_name(filename)}, True


def list_candidates():
    with connect() as db:
        return [dict(r) for r in db.execute('SELECT id,filename,sha256,filetype,created_at,length(content) AS size FROM candidates ORDER BY created_at DESC')]


def candidate_file(candidate_id):
    with connect() as db:
        row=db.execute('SELECT id,filename,sha256,filetype,content FROM candidates WHERE id=?', (candidate_id,)).fetchone()
        return dict(row) if row else None


def delete_candidate(candidate_id):
    """Also purge historic reports containing the deleted applicant's details."""
    with connect() as db:
        result=db.execute('DELETE FROM candidates WHERE id=?', (candidate_id,))
        if result.rowcount:
            for row in db.execute('SELECT id,report_json FROM runs').fetchall():
                doc=json.loads(row['report_json'])
                before=len(doc.get('candidates', []))
                doc['candidates']=[c for c in doc.get('candidates', []) if c.get('candidate_id')!=candidate_id]
                if len(doc['candidates']) != before:
                    from studio_core import report_statistics, report_suggestions
                    doc['statistics']=report_statistics(doc['candidates'])
                    doc['suggestions']=report_suggestions(doc['candidates'])
                    db.execute('UPDATE runs SET report_json=? WHERE id=?',
                               (json.dumps(doc,ensure_ascii=False),row['id']))
        return bool(result.rowcount)


def save_run(job_id, report):
    identifier=str(uuid.uuid4())
    with connect() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?)',
                   (identifier, job_id, now(), json.dumps(report, ensure_ascii=False)))
    return identifier


def list_runs(job_id=None):
    sql='SELECT runs.id,runs.job_id,runs.created_at,jobs.title AS job_title, length(runs.report_json) as report_size FROM runs JOIN jobs ON jobs.id=runs.job_id'
    params=[]
    if job_id:
        sql+=' WHERE runs.job_id=?';params=[job_id]
    with connect() as db:
        return [dict(r) for r in db.execute(sql+' ORDER BY runs.created_at DESC',params)]


def get_run(run_id):
    with connect() as db:
        row=db.execute('SELECT report_json FROM runs WHERE id=?',(run_id,)).fetchone()
        return json.loads(row['report_json']) if row else None


def delete_run(run_id):
    with connect() as db:
        db.execute('DELETE FROM runs WHERE id=?',(run_id,))


def _vacuum():
    """Rewrite the database file so deleted CV bytes do not linger in free pages."""
    db = sqlite3.connect(str(db_file()), timeout=30, isolation_level=None)
    try:
        db.execute('VACUUM')
    finally:
        db.close()


def delete_all_candidates():
    """Delete every CV. Saved shortlists are deleted too: they only hold candidate
    results, so with every candidate gone they would be empty. Jobs are kept."""
    with connect() as db:
        removed = db.execute('DELETE FROM candidates').rowcount
        db.execute('DELETE FROM runs')
    _vacuum()
    return removed


def reset_workspace():
    """Delete every CV, job and shortlist, then compact the database file."""
    with connect() as db:
        counts = {t: db.execute(f'DELETE FROM {t}').rowcount for t in ('runs', 'candidates', 'jobs')}
    _vacuum()
    return counts


def all_counts():
    with connect() as db:
        return {
            'jobs': db.execute('SELECT count(*) FROM jobs').fetchone()[0],
            'candidates': db.execute('SELECT count(*) FROM candidates').fetchone()[0],
            'runs': db.execute('SELECT count(*) FROM runs').fetchone()[0],
        }

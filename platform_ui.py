"""AI Resume Shortlist System: run `streamlit run platform_ui.py`.

Single-user localhost UI, no authentication. Resumes stay in .studio_data/workspace.sqlite3;
they are sent to NVIDIA only when the user picks AI scoring and ticks the consent box.
Three pages: Shortlist (the main flow), Job (one job description) and Resumes (up to 20).
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.extractor import ExtractionError, detect_type, extract_document
from modules.ai_evaluator import AIEvalOptions, ai_evaluate_batch, rubric_key
from modules.llm_engine import CloudClient, cloud_base_url, get_cloud_api_key
from modules.pipeline import InputError, evaluate_batch
from studio_core import build_report, ensure_valid_cv, export_csv, export_json, unpack_files, render_pdf_page
import studio_store as store

APP_NAME = 'AI Resume Shortlist'

st.set_page_config(page_title=APP_NAME, page_icon='🎯', layout='centered', initial_sidebar_state='collapsed')

NAV = ['Shortlist', 'Job', 'Resumes']
MAX_RESUMES = 20
CATEGORIES = ['Highly Suitable', 'Suitable', 'Moderate', 'Not Suitable']
CARDS_PER_PAGE = 10
MODES = {'ai': 'AI (NVIDIA)', 'rules': 'Quick, no AI'}

STYLES = r"""
<style>
/* Colours work on both the light and the dark theme: text inherits the theme colour,
   greys are translucent, and the category colours are mid-tones on a translucent tint. */
:root{--line:rgba(128,128,128,.22);--soft:rgba(128,128,128,.12);--brand:#5E9A86;
  --hi:#3E9A6B;--ok:#4C9A88;--mid:#C08A2B;--low:#CF6450;}
.block-container{padding-top:1.6rem;padding-bottom:4rem;max-width:860px;}
[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none;}
h1,h2,h3,h4{letter-spacing:-.01em;}
div.stButton>button,div.stDownloadButton>button,div.stFormSubmitButton>button{border-radius:10px;font-weight:500;white-space:nowrap;}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed var(--line);border-radius:14px;}
[data-testid="stExpander"] details{border-color:var(--line);border-radius:12px;}

.rs-brand{font-weight:600;font-size:1.05rem;}
.rs-brand span{color:var(--brand);}
.rs-title{font-size:1.45rem;font-weight:600;margin:6px 0 0;}
.rs-note{opacity:.62;font-size:.92rem;margin:2px 0 18px;}
.rs-h{font-size:1rem;font-weight:600;margin:22px 0 8px;}
.rs-line{opacity:.62;font-size:.88rem;margin:4px 0 12px;}

.rs-cand{display:flex;gap:14px;align-items:flex-start;}
.rs-rank{width:30px;height:30px;border-radius:50%;background:var(--soft);font-weight:600;display:grid;place-items:center;flex:none;font-size:.85rem;opacity:.8;}
.rs-name{font-weight:600;font-size:1rem;}
.rs-meta{opacity:.75;font-size:.86rem;margin:4px 0 8px;line-height:1.45;}
.rs-chips{display:flex;flex-wrap:wrap;gap:6px;}
.rs-chip{font-size:.74rem;padding:2px 9px;border-radius:999px;background:var(--soft);}
.rs-chip.no{background:rgba(207,100,80,.14);color:var(--low);}
.rs-chip.yes{background:rgba(62,154,107,.14);color:var(--hi);}

.rs-badge{display:inline-block;font-size:.7rem;font-weight:600;padding:2px 9px;border-radius:999px;margin-left:8px;vertical-align:2px;}
.rs-badge.hi{background:rgba(62,154,107,.15);color:var(--hi);}
.rs-badge.ok{background:rgba(76,154,136,.15);color:var(--ok);}
.rs-badge.mid{background:rgba(192,138,43,.15);color:var(--mid);}
.rs-badge.low{background:rgba(207,100,80,.15);color:var(--low);}
.rs-badge.out{background:var(--soft);opacity:.7;}

.rs-score{font-size:1.5rem;font-weight:600;text-align:right;font-variant-numeric:tabular-nums;}
.rs-score.big{font-size:2.2rem;}
.rs-score small{display:block;font-size:.72rem;font-weight:500;opacity:.6;}
.rs-score.hi{color:var(--hi)}.rs-score.ok{color:var(--ok)}.rs-score.mid{color:var(--mid)}.rs-score.low{color:var(--low)}.rs-score.out{opacity:.6}

.rs-req{display:flex;gap:10px;align-items:flex-start;padding:9px 0;border-bottom:1px solid var(--line);}
.rs-req:last-child{border-bottom:0;}
.rs-req .ic{width:20px;height:20px;border-radius:50%;display:grid;place-items:center;font-size:.7rem;font-weight:700;flex:none;margin-top:2px;}
.rs-req .ic.yes{background:rgba(62,154,107,.15);color:var(--hi);}
.rs-req .ic.no{background:rgba(207,100,80,.15);color:var(--low);}
.rs-req .body{flex:1;min-width:0;}
.rs-req .sk{font-weight:500;}
.rs-req .pc{opacity:.6;font-size:.8rem;white-space:nowrap;}
.rs-req .ev{opacity:.75;font-size:.82rem;margin-top:3px;font-style:italic;overflow-wrap:anywhere;}
.rs-req .ev.none{font-style:normal;opacity:.6;}
</style>
"""


def esc(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def percent(value):
    return '—' if value is None else f'{float(value):.0f}%'


def fmt(value):
    return f'{float(value or 0):g}'


def show(markup):
    st.markdown(markup, unsafe_allow_html=True)


def page_title(title, note=''):
    show(f'<div class="rs-title">{esc(title)}</div>' + (f'<div class="rs-note">{esc(note)}</div>' if note else ''))


def heading(text):
    show(f'<div class="rs-h">{esc(text)}</div>')


def band(category, status='scored'):
    """CSS class for a category: hi / ok / mid / low / out."""
    if status != 'scored':
        return 'out'
    return {'Highly Suitable': 'hi', 'Suitable': 'ok', 'Moderate': 'mid'}.get(category, 'low')


def badge(c):
    if c['status'] == 'duplicate':
        label = 'Duplicate'
    elif c['status'] != 'scored':
        label = 'Excluded'
    else:
        label = c['category']
    return f'<span class="rs-badge {band(c["category"], c["status"])}">{esc(label)}</span>'


def score_box(c, big=False):
    cls = band(c['category'], c['status'])
    text = '—' if c['score'] is None else f'{float(c["score"]):.0f}%'
    note = 'AI score' if c.get('ai') else 'rules score'
    return f'<div class="rs-score {cls}{" big" if big else ""}">{text}<small>{note}</small></div>'


def switch(page, **state):
    for key, value in state.items():
        st.session_state[key] = value
    st.session_state['nav_target'] = page
    st.rerun()


def confirm_delete(container, label, message, key):
    """Popover with a typed confirmation. Returns True when the delete is confirmed."""
    with container.popover(label, width='stretch'):
        st.write(message)
        typed = st.text_input('Type DELETE (in capitals) to confirm', key=key + '_typed')
        return st.button('Delete permanently', key=key + '_go', type='primary',
                         disabled=typed.strip() != 'DELETE')


def forget_deleted_data():
    """Drop in-memory copies (PDF previews, cached AI answers) and screen state for deleted data."""
    from modules import ai_evaluator, llm_engine
    st.cache_data.clear()
    ai_evaluator._cache.clear()
    llm_engine.clear_cache()
    for key in [k for k in st.session_state if k.endswith('_typed') or k.startswith('page_')]:
        del st.session_state[key]
    for key in ('detail', 'card_limit'):
        st.session_state.pop(key, None)


def cloud_key() -> str:
    """NVIDIA / OpenAI-compatible key from the environment or .streamlit/secrets.toml (never displayed)."""
    key = get_cloud_api_key()
    if not key:
        try:
            key = str(st.secrets.get('LLM_API_KEY', '')).strip()
        except Exception:
            key = ''
    return key


@st.cache_data(ttl=600, show_spinner=False)
def cloud_models(key: str, base_url: str):
    return CloudClient.list_cloud_models(key, base_url)


@st.cache_data(max_entries=24, show_spinner=False)
def pdf_page(raw: bytes, page: int, scale: float = 1.5):
    return render_pdf_page(raw, page, scale)


def current_job():
    """The one job the app works with: the most recently saved."""
    jobs = store.list_jobs()
    return jobs[0] if jobs else None


def load_demo():
    root = Path(__file__).resolve().parent / 'examples'
    jd = (root / 'Demo_Backend_Job.txt').read_text(encoding='utf-8')
    if not current_job():
        store.create_job('Demo · Senior Backend Engineer', jd)
    loaded = 0
    for sample in ('Jane_Backend.txt', 'Noah_Fullstack.txt', 'Eva_Analyst.txt', 'Arun_Junior.txt'):
        _, created = store.insert_candidate(sample, (root / sample).read_bytes())
        loaded += int(created)
    return loaded


def ai_gaps(ai):
    """Mandatory requirements the AI did not find in the resume."""
    return [r['requirement'] for r in ai.get('requirements', [])
            if r['importance'] == 'must' and r['status'] in ('not_evidenced', 'confirmed_gap')]


def preview_file(file):
    if not file:
        st.info('The original resume is no longer in your library.')
        return
    if file['filetype'] == 'pdf':
        try:
            _, total = pdf_page(file['content'], 1)
            page = 1
            if total > 1:
                page = st.number_input(f'Page (of {total})', min_value=1, max_value=total, value=1, step=1,
                                       key='page_' + file['id'])
            image, _ = pdf_page(file['content'], int(page))
            st.image(image, width='stretch')
        except Exception as exc:
            st.error(f'PDF preview unavailable ({type(exc).__name__}). Download the original file instead.')
    else:
        try:
            text = extract_document(file['content'], file['filetype']).text
            st.text_area('Resume text', value=text[:24000], height=520, disabled=True)
            if len(text) > 24000:
                st.caption('Preview shows the first 24,000 characters.')
        except ExtractionError as exc:
            st.warning(str(exc))


def scoring_options(candidates):
    """How to score and which resumes. Returns (chosen ids, {'mode','model','key','base_url','consent','ready'})."""
    key = cloud_key()
    ai = {'mode': 'rules', 'model': '', 'key': '', 'base_url': '', 'consent': False, 'ready': True}
    modes = list(MODES) if key else ['rules']
    mode = st.segmented_control('Scoring', modes, format_func=MODES.get, default=modes[0], key='score_mode')
    ai['mode'] = mode or modes[0]
    if not key:
        st.caption('AI scoring is off: add `LLM_API_KEY = "nvapi-…"` to `.streamlit/secrets.toml` and restart.')
    all_ids = [x['id'] for x in candidates]
    with st.expander('More options'):
        chosen = st.multiselect(f'Resumes to include (up to {MAX_RESUMES})', options=all_ids,
                                default=all_ids[:MAX_RESUMES], max_selections=MAX_RESUMES,
                                format_func=lambda cid: next(x['filename'] for x in candidates if x['id'] == cid))
        model = ''
        if ai['mode'] == 'ai':
            base_url = cloud_base_url(key)
            names = cloud_models(key, base_url)
            default = CloudClient.default_cloud_model(names, base_url)
            options = names if default in names else [default] + names
            model = st.selectbox('AI model', options, index=options.index(default), key='cloud_model')
            ai.update(key=key, base_url=base_url)
    if ai['mode'] == 'ai':
        consent = st.checkbox('OK to send these resumes to NVIDIA (emails, phone numbers and links are removed first)',
                              key='cloud_consent')
        ai.update(model=model, consent=consent, ready=consent)
        st.caption('Takes about 1–2 minutes per resume, a few at a time.')
    else:
        st.caption('Built-in rules only: takes a few seconds, nothing leaves this computer.')
    return chosen, ai


def shortlist_screen():
    detail = st.session_state.get('detail')
    if detail:
        report = store.get_run(detail['run'])
        cand = next((c for c in (report or {}).get('candidates', []) if c['uid'] == detail['uid']), None)
        if cand:
            candidate_detail(report, cand)
            return
        st.session_state['detail'] = None

    page_title('Shortlist', 'Rank your resumes against the job description.')
    job = current_job()
    candidates = store.list_candidates()
    if not job or not candidates:
        st.info('To start, add the job description and upload some resumes.')
        a, b, c = st.columns(3)
        if not job and a.button('Add the job', type='primary', width='stretch'):
            switch('Job')
        if not candidates and b.button('Upload resumes', type='primary' if job else 'secondary', width='stretch'):
            switch('Resumes')
        if not job and not candidates and c.button('Try with sample data', width='stretch'):
            loaded = load_demo()
            st.toast(f'Sample job and {loaded} sample resumes added.', icon='✅')
            st.rerun()
        return

    jid = job['id']
    with st.container(border=True):
        show(f'<div class="rs-name">{esc(job["title"])}</div>')
        chosen, ai = scoring_options(candidates)
        label = (f'Shortlist {len(chosen)} resume(s)' if ai['ready'] or ai['mode'] != 'ai'
                 else 'Tick the box above to start')
        if st.button(label, type='primary', width='stretch',
                     disabled=not chosen or not ai['ready']):
            if len(chosen) > MAX_RESUMES:
                st.error(f'You can shortlist up to {MAX_RESUMES} resumes at a time.')
            else:
                run_matching(jid, job, chosen, ai)

    runs = store.list_runs(jid)
    if not runs:
        st.caption('No shortlist yet.')
        return
    runid = runs[0]['id']
    report = store.get_run(runid)
    if report:
        results(report, runid)


def previous_rubric(jid, job):
    """The AI rubric from this job's last AI run, if the job text is unchanged, so runs stay comparable."""
    key = rubric_key(job['title'], job['description'])
    for run in store.list_runs(jid)[:5]:
        rubric = ((store.get_run(run['id']) or {}).get('ai_summary') or {}).get('rubric')
        if rubric and rubric.get('key') == key:
            return rubric
    return None


def run_matching(jid, job, chosen, ai):
    progress = st.progress(0, text='Reading resumes…')
    selected = [store.candidate_file(cid) for cid in chosen]
    files = [(x['filename'], x['content']) for x in selected if x]
    hashes = {x['sha256']: x['id'] for x in selected if x}

    def on_progress(done, total, message):
        progress.progress(min(1.0, done / max(1, total)), text=f'{message} ({done} of {total})')
    try:
        with st.spinner('Reading resumes…'):
            result = evaluate_batch(job['description'], files, progress=on_progress)
        if ai['mode'] == 'ai':
            progress.progress(0.0, text='Sending resumes to the AI…')
            with st.spinner('The AI is reading each resume. This can take a few minutes…'):
                ai_evaluate_batch(result, job['description'], job['title'],
                                  AIEvalOptions(model=ai['model'], api_key=ai['key'], base_url=ai['base_url'],
                                                consent=ai['consent']),
                                  progress=on_progress, rubric=previous_rubric(jid, job))
        report = build_report(result, jid, job['title'], hashes)
        store.save_run(jid, report)
        progress.empty()
        st.toast(f"Shortlist ready: {report['statistics']['scored']} resumes ranked.", icon='🎯')
        st.session_state['card_limit'] = CARDS_PER_PAGE
        st.rerun()
    except (InputError, ValueError, OSError) as exc:
        progress.empty()
        st.error(str(exc))


def candidate_card(c, runid, key):
    """One ranked candidate: rank, name, category, a short summary and the score."""
    ai = c.get('ai') or {}
    if c['status'] == 'scored' and ai:
        summary = ai.get('summary') or ''
        meta = esc(summary[:200]) + ('…' if len(summary) > 200 else '')
        gaps = ai_gaps(ai)
    elif c['status'] == 'scored':
        meta = (f'{c["matched_requirements"]} of {c["requirement_count"]} requirements found · '
                f'{fmt(c["relevant_years"])} yrs relevant experience')
        gaps = c['missing_must_haves']
    else:
        meta, gaps = esc(c['reject_reason'] or c['status']), []
    chips = ''.join(f'<span class="rs-chip no">Missing: {esc(s)}</span>' for s in gaps[:3])
    with st.container(border=True):
        left, right = st.columns([5, 1.1], vertical_alignment='center')
        left.markdown(
            f'<div class="rs-cand"><div class="rs-rank">{c.get("rank") or "—"}</div><div>'
            f'<div class="rs-name">{esc(c["display_name"])}{badge(c)}</div>'
            f'<div class="rs-meta">{meta}</div><div class="rs-chips">{chips}</div></div></div>',
            unsafe_allow_html=True)
        with right:
            show(score_box(c))
            if st.button('Open', key=f'open_{key}', width='stretch'):
                switch('Shortlist', detail={'run': runid, 'uid': c['uid']})


def results(report, runid):
    stats = report['statistics']
    head, csv_col, json_col = st.columns([4, 1, 1], vertical_alignment='bottom')
    with head:
        heading('Results')
    csv_col.download_button('CSV', data=export_csv(report), file_name=f'shortlist_{runid[:8]}.csv',
                            mime='text/csv', width='stretch', key=f'csv_{runid}')
    json_col.download_button('JSON', data=export_json(report), file_name=f'shortlist_{runid[:8]}.json',
                             mime='application/json', width='stretch', key=f'json_{runid}')
    cloud = report.get('ai_summary') or {}
    how = f"scored by AI ({cloud.get('model')})" if cloud.get('succeeded') else 'scored by the built-in rules'
    show(f'<div class="rs-line">{stats["processed"]} resumes · '
         f'{stats["highly_suitable"] + stats["suitable"]} strong matches · {esc(how)}</div>')
    if cloud.get('enabled') and (cloud.get('errors') or cloud.get('skipped_injection')):
        reason = cloud['errors'][0] if cloud.get('errors') else 'resume contains instructions aimed at AI'
        if cloud.get('succeeded'):
            missed = sum(1 for x in report['candidates'] if x['status'] == 'not_evaluated')
            if missed:
                st.warning(f"{missed} resume(s) could not be evaluated by the AI, so they are not ranked "
                           f"(see 'Not scored'). Run the shortlist again to retry them. Reason: {reason}")
        else:
            st.warning(f"The AI evaluation did not run, so every resume was scored by the built-in rules. "
                       f"Reason: {reason}")

    scored = [x for x in report['candidates'] if x['status'] == 'scored']
    excluded = [x for x in report['candidates'] if x['status'] != 'scored']
    if scored:
        counts = {cat: sum(1 for x in scored if x['category'] == cat) for cat in CATEGORIES}
        pick = st.segmented_control('Category', ['All'] + [k for k in CATEGORIES if counts[k]], default='All',
                                    key='match_category', label_visibility='collapsed',
                                    format_func=lambda k: f'All ({len(scored)})' if k == 'All' else f'{k} ({counts[k]})')
        visible = [x for x in scored if pick in (None, 'All') or x['category'] == pick]
        limit = st.session_state.get('card_limit', CARDS_PER_PAGE)
        for i, cand in enumerate(visible[:limit]):
            candidate_card(cand, runid, key=f'{runid}_{i}')
        if len(visible) > limit:
            if st.button(f'Show more ({len(visible) - limit} left)', width='stretch'):
                st.session_state['card_limit'] = limit + CARDS_PER_PAGE
                st.rerun()
    else:
        st.info('None of the resumes could be scored. See the list below for reasons.')
    if excluded:
        with st.expander(f'Not scored ({len(excluded)})'):
            st.dataframe(pd.DataFrame([{'Resume': x['filename'], 'Why': x['reject_reason']} for x in excluded]),
                         hide_index=True, width='stretch')
    st.caption('Scores help you decide who to look at first. They support, not replace, your own review.')


def requirement_rows(ai):
    icon = {'met': ('yes', '✓', 'found'), 'partial': ('yes', '~', 'partly'), 'inferred': ('yes', '≈', 'likely'),
            'not_evidenced': ('no', '?', 'not stated'), 'confirmed_gap': ('no', '✗', 'not met'),
            'missing': ('no', '?', 'not stated')}
    order = {'met': 0, 'partial': 1, 'inferred': 2, 'not_evidenced': 3, 'missing': 3, 'confirmed_gap': 4}
    rows = []
    for r in sorted(ai['requirements'], key=lambda r: (order.get(r['status'], 3), r['importance'] != 'must')):
        cls, mark, label = icon.get(r['status'], icon['not_evidenced'])
        tag = ' <span class="rs-chip">must</span>' if r['importance'] == 'must' else ''
        ev = ''
        if r['evidence']:
            unverified = '' if r['evidence_verified'] else ' (not found word-for-word in the resume)'
            ev = f'<div class="ev">“{esc(r["evidence"][:300])}”{esc(unverified)}</div>'
        if r.get('note') and r['status'] in ('partial', 'inferred', 'confirmed_gap'):
            ev += f'<div class="ev none">{esc(r["note"][:300])}</div>'
        rows.append(f'<div class="rs-req"><div class="ic {cls}">{mark}</div><div class="body">'
                    f'<span class="sk">{esc(r["requirement"])}</span>{tag}{ev}</div>'
                    f'<div class="pc">{esc(label)}</div></div>')
    return ''.join(rows)


def rules_rows(reqs):
    rows = []
    for r in sorted(reqs, key=lambda r: (not r['matched'], r['importance'] != 'must')):
        tag = ' <span class="rs-chip">must</span>' if r['importance'] == 'must' else ''
        if r['matched']:
            ev = f'<div class="ev">“{esc(r["evidence"][:300])}”</div>' if r['evidence'] else ''
            rows.append(f'<div class="rs-req"><div class="ic yes">✓</div><div class="body">'
                        f'<span class="sk">{esc(r["skill"])}</span>{tag}{ev}</div>'
                        f'<div class="pc">found</div></div>')
        else:
            rows.append(f'<div class="rs-req"><div class="ic no">?</div><div class="body">'
                        f'<span class="sk">{esc(r["skill"])}</span>{tag}</div>'
                        f'<div class="pc">not found</div></div>')
    return ''.join(rows)


def candidate_detail(report, c):
    if st.button('← Back'):
        switch('Shortlist', detail=None)
    ai = c.get('ai') or {}
    with st.container(border=True):
        left, right = st.columns([4, 1.2], vertical_alignment='center')
        left.markdown(f'<div class="rs-name" style="font-size:1.3rem">{esc(c["display_name"])}{badge(c)}</div>'
                      f'<div class="rs-meta">{esc(c["filename"])} · {esc(report["job_title"])}'
                      + (f' · rank #{c["rank"]}' if c.get('rank') else '') + '</div>', unsafe_allow_html=True)
        with right:
            show(score_box(c, big=True))
    if c['reject_reason']:
        st.warning(c['reject_reason'])

    if ai:
        if ai.get('summary'):
            st.write(ai['summary'])
        if ai.get('cap_note'):
            st.caption(ai['cap_note'])
        a, b = st.columns(2, gap='large')
        for col, title, items in ((a, 'Strengths', ai.get('strengths')), (b, 'Concerns', ai.get('concerns'))):
            with col:
                heading(title)
                for item in items or ['—']:
                    st.write('• ' + item)
        if ai.get('requirements'):
            with st.expander('Requirements checked'):
                show(requirement_rows(ai))
    elif c['status'] == 'scored':
        show(f'<div class="rs-line">{c["matched_requirements"]} of {c["requirement_count"]} requirements found · '
             f'{fmt(c["relevant_years"])} yrs relevant · {fmt(c["total_years"])} yrs total</div>')
        if c['requirements']:
            with st.expander('Requirements checked', expanded=True):
                show(rules_rows(c['requirements']))

    questions = list(ai.get('interview_questions') or [])
    if not ai and c['missing_must_haves']:
        questions.append('Ask about: ' + ', '.join(c['missing_must_haves']))
    if questions:
        with st.expander('Questions to ask in the interview'):
            for q in questions:
                st.write('• ' + q)
    with st.expander('Experience & education'):
        st.caption(f"{fmt(c['total_years'])} yrs total · {fmt(c['relevant_years'])} yrs relevant to this job")
        if c['roles']:
            st.dataframe(pd.DataFrame([{'Role': r['role'], 'Period': r['period'], 'Months': r['months'],
                                        'Relevant': '✓' if r['relevant'] else ''} for r in c['roles']]),
                         hide_index=True, width='stretch')
        for degree in c['degrees']:
            st.write('• ' + degree)
    with st.expander('Resume'):
        original = store.candidate_file(c['candidate_id']) if c['candidate_id'] else None
        if original:
            st.download_button('Download original', data=original['content'], file_name=original['filename'],
                               mime='application/octet-stream')
        preview_file(original)
    if ai:
        with st.expander('How the score was worked out'):
            st.caption(f'AI model: {ai.get("model", "")}'
                       + (f' · confidence: {ai["confidence"]}' if ai.get('confidence') else '')
                       + f' · built-in rules score: {percent(c.get("rule_score"))}')
            for item in ai.get('score_reasons') or []:
                st.write('• ' + item)


def new_job_form():
    with st.form('new_job', clear_on_submit=False, border=False):
        title = st.text_input('Job title', placeholder='e.g. Senior Backend Engineer')
        description = st.text_area('Job description', height=220,
                                   placeholder='Paste the full job description: skills, years of experience, education…')
        jd_file = st.file_uploader('…or upload it (PDF, DOCX, TXT)', type=['pdf', 'docx', 'txt'], key='jd_upload')
        submit = st.form_submit_button('Save job', type='primary')
    if submit:
        try:
            text = description.strip()
            if jd_file and not text:
                text = extract_document(jd_file.getvalue(), detect_type(jd_file.name)).text
            store.create_job(title, text)
            st.toast('Job saved.', icon='✅')
            st.rerun()
        except (ValueError, InputError, ExtractionError) as exc:
            st.error(str(exc))


def job_screen():
    page_title('Job', 'Resumes are scored against this job description.')
    job = current_job()
    if not job:
        with st.container(border=True):
            new_job_form()
        return
    l, m, r = st.columns([1.6, 1, 1])
    if l.button('Shortlist resumes', type='primary', width='stretch'):
        switch('Shortlist', detail=None)
    with m.popover('Edit', width='stretch'):
        with st.form('edit_job', border=False):
            name = st.text_input('Title', value=job['title'])
            desc = st.text_area('Description', value=job['description'], height=260)
            if st.form_submit_button('Save changes', type='primary'):
                try:
                    store.update_job(job['id'], name, desc)
                    st.toast('Changes saved. The last shortlist keeps its original scores.', icon='✅')
                    st.rerun()
                except (ValueError, InputError) as exc:
                    st.error(str(exc))
    with r.popover('Delete', width='stretch'):
        st.write('Deletes this job and its shortlists so you can add a different one. Resumes are kept.')
        if st.button('Delete permanently', key='del_job_' + job['id'], type='primary'):
            store.delete_job(job['id'])
            st.rerun()
    with st.container(border=True):
        show(f'<div class="rs-name">{esc(job["title"])}</div>')
        st.text(job['description'])


def resumes_screen():
    page_title('Resumes', f'Upload up to {MAX_RESUMES} resumes.')
    room = MAX_RESUMES - len(store.list_candidates())
    incoming = None
    if room > 0:
        incoming = st.file_uploader('Drop resumes here', type=['pdf', 'docx', 'txt', 'zip'],
                                    accept_multiple_files=True,
                                    key=f'bulk_uploader_{st.session_state.get("upload_round", 0)}',
                                    help=f'PDF, DOCX, TXT, or ZIP files. Room for {room} more.')
    else:
        st.info(f'You have {MAX_RESUMES} resumes, the maximum. Delete some to add others.')
    last = st.session_state.pop('upload_result', None)
    if last:
        st.success(last['message'])
        if last['over']:
            st.warning(f"{last['over']} file(s) not added: the limit is {MAX_RESUMES} resumes.")
        if last['errors']:
            with st.expander(f"{len(last['errors'])} file(s) could not be added", expanded=True):
                for err in last['errors'][:60]:
                    st.write('• ' + err)
    if incoming and st.button(f'Add {len(incoming)} file(s)', type='primary'):
        pairs, errors = unpack_files([(x.name, x.getvalue()) for x in incoming])
        progress = st.progress(0, text='Checking files…')
        imported = duplicates = over = 0
        for idx, (name, raw) in enumerate(pairs):
            if imported >= room:
                over += 1
                continue
            try:
                ensure_valid_cv(name, raw)
                _, created = store.insert_candidate(name, raw)
                imported += int(created)
                duplicates += int(not created)
            except (ExtractionError, ValueError, OSError) as exc:
                errors.append(f'{name}: {exc}')
            progress.progress((idx + 1) / max(1, len(pairs)), text=f'Checked {idx + 1}/{len(pairs)}')
        progress.empty()
        st.session_state['upload_result'] = {
            'message': f'Added {imported} resume(s).' + (f' Skipped {duplicates} already added.' if duplicates else ''),
            'over': over, 'errors': errors}
        st.session_state['upload_round'] = st.session_state.get('upload_round', 0) + 1
        st.rerun()

    records = store.list_candidates()
    heading(f'Your resumes ({len(records)} of {MAX_RESUMES})')
    if not records:
        st.caption('No resumes yet. Drop some files above.')
    else:
        st.dataframe(pd.DataFrame([{'Resume': r['filename'], 'Type': r['filetype'].upper(),
                                    'Added': r['created_at'][:16].replace('T', ' ')} for r in records]),
                     hide_index=True, width='stretch', height=min(420, 38 + 35 * len(records)))
        with st.expander('Open a resume'):
            selected = st.selectbox('Resume', [r['id'] for r in records], label_visibility='collapsed',
                                    format_func=lambda x: next(r['filename'] for r in records if r['id'] == x))
            file = store.candidate_file(selected)
            l, r, _ = st.columns([1, 1, 2])
            l.download_button('Download', data=file['content'], file_name=file['filename'],
                              mime='application/octet-stream', width='stretch')
            with r.popover('Delete', width='stretch'):
                st.write('Removes this resume and its results from every saved shortlist.')
                if st.button('Delete permanently', key='del_cv_' + selected, type='primary'):
                    store.delete_candidate(selected)
                    st.rerun()
            preview_file(file)

    counts = store.all_counts()
    if any(counts.values()):
        heading('Start fresh')
        a, b, _ = st.columns([1, 1, 1])
        if records and confirm_delete(a, 'Delete all resumes',
                                      f'Deletes **all {len(records)} resume(s)** and every saved shortlist. '
                                      'The job is kept. This cannot be undone.', key='delete_all_cvs'):
            removed = store.delete_all_candidates()
            forget_deleted_data()
            st.toast(f'{removed} resume(s) deleted.', icon='🗑')
            st.rerun()
        if confirm_delete(b, 'Reset everything',
                          f"Deletes **all {counts['candidates']} resume(s), {counts['jobs']} job(s) and "
                          f"{counts['runs']} shortlist(s)**. This cannot be undone. Downloaded CSV/JSON files "
                          "are not affected.", key='reset_all'):
            removed = store.reset_workspace()
            forget_deleted_data()
            st.toast(f"Cleared: {removed['candidates']} resumes, {removed['jobs']} jobs, "
                     f"{removed['runs']} shortlists.", icon='🗑')
            st.rerun()


def top_nav():
    """Brand on the left, three page buttons on the right. Returns the current page."""
    page = st.session_state.get('nav_menu', NAV[0])
    if page not in NAV:
        page = NAV[0]
    brand, *cols = st.columns([2.2, 1, 1, 1], vertical_alignment='center')
    brand.markdown('<div class="rs-brand">AI Resume <span>Shortlist</span></div>', unsafe_allow_html=True)
    for col, name in zip(cols, NAV):
        if col.button(name, key=f'nav_{name}', width='stretch',
                      type='primary' if name == page else 'tertiary'):
            st.session_state['nav_menu'] = name
            st.session_state['detail'] = None
            st.rerun()
    return page


def main():
    store.initialize()
    target = st.session_state.pop('nav_target', None)
    if target in NAV:
        st.session_state['nav_menu'] = target
    show(STYLES)
    page = top_nav()
    if page != 'Shortlist':
        st.session_state['detail'] = None
    {'Shortlist': shortlist_screen, 'Job': job_screen, 'Resumes': resumes_screen}[page]()


if __name__ == '__main__':
    main()

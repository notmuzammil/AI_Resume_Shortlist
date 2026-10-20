"""Local single-user recruiter workflow and safe, explainable presentation data.

The deterministic parser/matcher in modules/ remains the source of truth. No
recruitment scores are inferred from a UI bar chart or from an LLM.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import PurePosixPath

from config import EXP_DEPTH_YEARS, MAX_FILE_BYTES, MAX_FILES_PER_BATCH
from modules.extractor import ExtractionError, extract_document, detect_type
from modules.pipeline import BatchResult, Candidate

ALLOWED = {'.pdf', '.docx', '.txt'}
ZIP_LIMIT = min(MAX_FILES_PER_BATCH * MAX_FILE_BYTES, 150 * 1_048_576)


def clean_name(filename: str) -> str:
    """Normalize filenames for display/storage; do not use untrusted paths."""
    name = str(filename or '').replace('\\', '/').split('/')[-1]
    name = ''.join(ch for ch in name if ch.isprintable() and ch not in '\x00\r\n')
    return name[:220] or 'unnamed.txt'


def unpack_files(items: list[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Accept multi-file uploads and optional ZIPs, with bounded decompression.

    Invalid items are reported, not silently accepted. The extraction engine
    also validates file signatures, page counts, and DOCX compression.
    """
    ready, errors, total = [], [], 0
    for incoming_name, data in items:
        name = clean_name(incoming_name)
        if name.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for item in archive.infolist():
                        if item.is_dir() or item.filename.startswith('__MACOSX/'):
                            continue
                        if item.flag_bits & 1:
                            errors.append(f'{name}: encrypted ZIP entry skipped')
                            continue
                        entry = PurePosixPath(item.filename)
                        if '..' in entry.parts or item.filename.startswith(('/', '\\')):
                            errors.append(f'{name}: unsafe ZIP entry skipped')
                            continue
                        basename = clean_name(item.filename)
                        if not any(basename.lower().endswith(ext) for ext in ALLOWED):
                            continue
                        if item.file_size > MAX_FILE_BYTES:
                            errors.append(f'{basename}: exceeds per-file limit')
                            continue
                        if total + item.file_size > ZIP_LIMIT or len(ready) >= MAX_FILES_PER_BATCH:
                            errors.append('Batch size limit reached; remaining ZIP entries skipped')
                            break
                        with archive.open(item) as part:
                            raw = part.read(MAX_FILE_BYTES + 1)
                        if len(raw) > MAX_FILE_BYTES:
                            errors.append(f'{basename}: file too large')
                            continue
                        ready.append((basename, raw))
                        total += len(raw)
            except (zipfile.BadZipFile, RuntimeError, ValueError, OSError) as exc:
                errors.append(f'{name}: cannot read ZIP ({type(exc).__name__})')
        elif any(name.lower().endswith(ext) for ext in ALLOWED):
            if len(data) > MAX_FILE_BYTES:
                errors.append(f'{name}: exceeds per-file limit')
            elif len(ready) >= MAX_FILES_PER_BATCH or total + len(data) > ZIP_LIMIT:
                errors.append(f'{name}: batch limit reached')
            else:
                ready.append((name, data))
                total += len(data)
        else:
            errors.append(f'{name}: unsupported file type')
    return ready, errors


def ensure_valid_cv(name: str, raw: bytes):
    res = extract_document(raw, detect_type(name))
    if len(res.text.strip()) < 30:
        raise ExtractionError('Insufficient readable CV text; check OCR availability')
    return res


def safe_score(value) -> float:
    try:
        v = float(value)
        return round(min(100.0, max(0.0, v)), 2) if math.isfinite(v) else 0.0
    except (ValueError, TypeError):
        return 0.0


def display_name(filename: str) -> str:
    """Resume filename is a label, not a reliable person-name extraction."""
    stem = re.sub(r'(?i)\.(pdf|docx|txt)$', '', clean_name(filename))
    return re.sub(r'[_-]+', ' ', stem).strip()[:120] or 'Candidate'


def component_data(c: Candidate):
    return {key: {'percent': safe_score(comp.value * 100) if comp.available else None,
                  'available': bool(comp.available), 'detail': comp.detail}
            for key, comp in c.components.items()}


def requirement_data(c: Candidate):
    rows = []
    for req, quality, best, hits in c.requirement_status:
        rows.append({
            'skill': req.label, 'importance': req.importance,
            'match_percent': safe_score(quality * 100),
            'matched': quality > 0, 'match_type': best.match_type if best else 'missing',
            'confidence': best.confidence if best else '',
            'hands_on': bool(best and best.hands_on),
            'evidence': (best.evidence or '')[:1200] if best else '',
            'basis': (best.basis or '')[:500] if best else '',
            'via': (best.via or '')[:120] if best else '',
            'source': (req.source or '')[:350],
        })
    return rows


def experience_meter(years: float, required: float) -> dict:
    """Comparison metric only; not another weighted scoring component."""
    baseline = float(required) if required > 0 else float(EXP_DEPTH_YEARS)
    return {'years': float(years), 'reference_years': baseline,
            'reference_type': 'JD minimum' if required > 0 else 'display reference (no explicit JD minimum)',
            'percent': safe_score(years / max(baseline, 0.01) * 100)}


def normalize_category(category: str) -> str:
    return next((name for name in ('Highly Suitable', 'Not Suitable', 'Suitable', 'Moderate')
                 if category.endswith(name)), category)


def candidate_data(c: Candidate, required_years: float = 0.0, candidate_id: str = '') -> dict:
    requirements = requirement_data(c)
    matching = sum(row['matched'] for row in requirements)
    return {
        'candidate_id': candidate_id,
        'uid': c.uid, 'filename': clean_name(c.name), 'display_name': display_name(c.name),
        'rank': c.rank, 'status': c.status, 'category': normalize_category(c.category),
        'score': safe_score(c.score) if c.status == 'scored' else None,
        'raw_score': safe_score(c.raw_score),
        'reject_reason': c.reject_reason,
        'total_years': c.total_years, 'relevant_years': c.relevant_years,
        'total_experience': experience_meter(c.total_years, required_years),
        'relevant_experience': experience_meter(c.relevant_years, required_years),
        'degrees': [str(d.display()) for d in c.degrees],
        'degrees_display': c.degrees_display,
        'components': component_data(c), 'requirements': requirements,
        'matched_requirements': matching, 'requirement_count': len(requirements),
        'skills_coverage_percent': safe_score(100 * matching / len(requirements)) if requirements else None,
        'missing_must_haves': [r['skill'] for r in requirements if r['importance'] == 'must' and not r['matched']],
        'notes': list(c.notes), 'warnings': list(c.warnings), 'flags': list(c.flags),
        'titles': list(c.titles), 'llm_used': c.llm_used, 'llm_rejections': list(c.llm_rejections),
        'score_source': 'ai' if c.ai else 'rules',
        'ai': dict(c.ai) if c.ai else None,
        'rule_score': safe_score(c.rule_score) if c.rule_score is not None else (
            safe_score(c.score) if c.status == 'scored' else None),
        'rule_category': normalize_category(c.rule_category or c.category),
        'roles': [ {'period': role.label(), 'months': role.months, 'role': role.header[:200],
                    'relevant': bool(role.relevant), 'reason': role.relevance_reason}
                   for role in c.roles ],
        'undated_roles': [{'role': r.header[:200], 'relevant': bool(r.relevant), 'reason': r.relevance_reason}
                          for r in c.undated_roles],
    }


def jd_data(profile) -> dict:
    return {'title': profile.title, 'seniority': profile.seniority,
            'required_years': profile.required_years,
            'effective_required_years': profile.effective_required_years,
            'bachelors_required': profile.bachelors_required,
            'graduates_only': profile.graduates_only,
            'education_fields': sorted(profile.education_fields),
            'warnings': list(profile.warnings),
            'requirements': [{'skill': req.label, 'importance': req.importance,
                              'source': req.source} for req in profile.requirements]}


def build_report(result: BatchResult, job_id: str, job_title: str, ids_by_sha: dict[str, str]) -> dict:
    candidates = [candidate_data(c, result.profile.required_years, ids_by_sha.get(c.sha256, ''))
                  for c in result.candidates]
    report = {'schema_version': 1, 'job_id': job_id, 'job_title': job_title,
              'generated_at': datetime.now(timezone.utc).isoformat(),
              'duration_seconds': result.seconds, 'jd': jd_data(result.profile),
              'warnings': list(result.warnings), 'llm_summary': result.llm_summary,
              'ai_summary': dict(getattr(result, 'ai_summary', None) or {}),
              'candidates': candidates}
    report['statistics'] = report_statistics(candidates)
    report['suggestions'] = report_suggestions(candidates)
    return report


def report_statistics(candidates: list[dict]) -> dict:
    scored = [x for x in candidates if x['status'] == 'scored']
    rejected = [x for x in candidates if x['status'] == 'rejected']
    duplicate = [x for x in candidates if x['status'] == 'duplicate']
    not_evaluated = [x for x in candidates if x['status'] == 'not_evaluated']
    counts = Counter(x['category'] for x in scored)
    scores = [x['score'] for x in scored if x['score'] is not None]
    reqs = Counter()
    for c in scored:
        for r in c['requirements']:
            if not r['matched']:
                reqs[r['skill']] += 1
    return {'processed': len(candidates), 'scored': len(scored), 'rejected': len(rejected),
            'duplicates': len(duplicate), 'not_evaluated': len(not_evaluated), 'highly_suitable': counts.get('Highly Suitable', 0),
            'suitable': counts.get('Suitable', 0), 'moderate': counts.get('Moderate', 0),
            'not_suitable': counts.get('Not Suitable', 0),
            'average_score': safe_score(sum(scores)/len(scores)) if scores else None,
            'highest_score': max(scores) if scores else None,
            'top_missing_skills': [{'skill': s, 'count': n} for s, n in reqs.most_common(8)]}


def report_suggestions(candidates: list[dict]) -> list[str]:
    stats = report_statistics(candidates)
    tips = []
    if not stats['scored']:
        return ['No CVs were successfully scored. Inspect rejected documents and OCR setup.']
    if stats['highly_suitable'] == 0:
        tips.append('No candidates reached the Highly Suitable band; review requirements with a recruiter and broaden sourcing if appropriate.')
    if stats['rejected']:
        tips.append(f"{stats['rejected']} CV(s) were rejected by parsing or eligibility rules; inspect each reason before excluding applicants.")
    if stats['top_missing_skills']:
        top = stats['top_missing_skills'][0]
        tips.append(f"Most frequently missing requirement: {top['skill']} ({top['count']} scored CVs). Verify it is essential to the role.")
    if any(c['missing_must_haves'] for c in candidates if c['status']=='scored'):
        tips.append('Review missing must-haves individually; inferred skills are not equivalent to verified hands-on experience.')
    tips.append('Treat percentages as JD-fit heuristics, not hiring probabilities. Verify qualifications and use consistent interviews before decisions.')
    return tips


def csv_escape(value):
    value = '' if value is None else str(value)
    return "'"+value if value[:1] in '=+-@\t\r\n' else value


def export_csv(report: dict) -> bytes:
    out = io.StringIO(newline='')
    fields = ['rank', 'candidate', 'file', 'match_percent', 'category', 'status', 'scored_by', 'rules_percent',
              'ai_summary', 'skills_percent',
              'relevant_experience_years', 'total_experience_years', 'skills_matched',
              'skills_count', 'missing_must_haves', 'notes']
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for c in report['candidates']:
        ai = c.get('ai') or {}
        writer.writerow({k: csv_escape(v) for k,v in {
            'rank': c['rank'], 'candidate': c['display_name'], 'file': c['filename'],
            'match_percent': c['score'], 'category': c['category'], 'status': c['status'],
            'scored_by': 'AI (' + ai.get('model', '') + ')' if ai else 'rules',
            'rules_percent': c.get('rule_score'), 'ai_summary': ai.get('summary', ''),
            'skills_percent': c['components'].get('skills', {}).get('percent'),
            'relevant_experience_years': c['relevant_years'], 'total_experience_years': c['total_years'],
            'skills_matched': c['matched_requirements'], 'skills_count': c['requirement_count'],
            'missing_must_haves': '; '.join(c['missing_must_haves']),
            'notes': '; '.join(c['notes'][:3]),
        }.items()})
    return out.getvalue().encode('utf-8-sig')


def export_json(report: dict) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')


def render_pdf_page(raw: bytes, page: int, scale: float=1.5) -> tuple[bytes, int]:
    """Convert a requested page to PNG bytes without embedding active PDF content."""
    import pypdfium2 as pdfium
    doc=pdfium.PdfDocument(raw)
    try:
        total=len(doc)
        if page < 1 or page > total:
            raise ValueError('Invalid PDF page')
        image=doc[page-1].render(scale=scale).to_pil()
        buf=io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue(), total
    finally:
        doc.close()

"""
End-to-end evaluation pipeline, independent of any UI.

    evaluate_batch(jd_text, [(filename, bytes), ...], llm=LLMOptions(...))

1. Resolve the JD once (jd_profile.build_profile).
2. Validate + extract every CV (bad files are reported, never crash the batch).
3. Detect duplicate uploads (same bytes) — scored once, flagged.
4. Deterministic analysis per CV: degrees, experience (total + relevant),
   skills, requirement matching, hard gates, score.
5. Optional local-LLM pass for borderline candidates' UNMATCHED
   requirements, with a time budget and a circuit breaker; failures fall
   back to the deterministic result and are reported.
6. Deterministic ranking: score desc, then file name (ties never depend
   on thread completion order).
"""
import hashlib
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

from config import (
    BELOW_MIN_EXPERIENCE_CAP, BELOW_MIN_EXPERIENCE_FLOOR_CAP, BELOW_MIN_EXPERIENCE_POLICY,
    BELOW_MIN_EXPERIENCE_SCALE_YEARS, MIN_EXPERIENCE_TOLERANCE_RATIO, MIN_EXPERIENCE_TOLERANCE_YEARS,
    MISSING_TITLE_SKILL_CAP,
    EXTRACTION_WORKERS, LLM_BATCH_BUDGET_SECONDS, LLM_CIRCUIT_BREAKER_FAILURES,
    LLM_CLOUD_PARALLEL, LLM_MAX_CANDIDATES, LLM_MIN_DETERMINISTIC_SCORE, LLM_PARALLEL, MATCH_CREDIT, MAX_CV_TEXT_CHARS,
    MAX_FILES_PER_BATCH, MAX_JD_CHARS,
    MIN_TEXT_CHARS, OVERQUALIFIED_MARGIN_YEARS, WEIGHTS,
)
from modules import scoring
from modules.education import (
    extract_degrees, extract_education, format_degrees, has_bachelors, highest_completed_level,
)
from modules.experience import analyse_experience, extract_job_titles, merged_years
from modules.extractor import ExtractionError, detect_type, extract_document, normalize_text
from modules.jd_profile import ROLE_FAMILIES, JDProfile, build_profile, title_tokens
from modules.llm_engine import (
    CloudClient, LLMError, LLMUnavailable, OllamaClient, detect_prompt_injection, get_cloud_api_key,
)
from modules.matcher import (
    CVIndex, SkillMatch, evidence_supports, is_concept_requirement, match_requirements,
)
from modules.skill_extractor import extract_skills

logger = logging.getLogger(__name__)


class InputError(ValueError):
    """Invalid batch input, with a user-safe message."""


@dataclass
class LLMOptions:
    enabled: bool = False
    model: str = ""
    budget_seconds: float = LLM_BATCH_BUDGET_SECONDS
    min_score: float = LLM_MIN_DETERMINISTIC_SCORE
    max_candidates: int = LLM_MAX_CANDIDATES
    provider: str = "ollama"
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    cloud_consent: bool = False


@dataclass
class Candidate:
    uid: str
    name: str
    sha256: str = ""
    status: str = "scored"
    reject_reason: str = ""
    score: float = 0.0
    raw_score: float = 0.0
    category: str = ""
    rank: int = None
    degrees: list = field(default_factory=list)
    degrees_display: str = ""
    still_enrolled: bool = False
    total_years: float = 0.0
    relevant_years: float = 0.0
    roles: list = field(default_factory=list)
    undated_roles: list = field(default_factory=list)
    titles: list = field(default_factory=list)
    cv_fields: set = field(default_factory=set)
    matches: dict = field(default_factory=dict)
    requirement_status: list = field(default_factory=list)
    components: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    llm_used: bool = False
    llm_rejections: list = field(default_factory=list)
    text: str = ""
    file_type: str = ""
    duplicate_of: str = ""
    ai: dict = field(default_factory=dict)
    rule_score: float = None
    rule_category: str = ""
    gate_rejected: bool = False


    @property
    def missing(self):
        return [(r.label, r.importance) for r, q, *_ in self.requirement_status if q <= 0]


@dataclass
class BatchResult:
    profile: JDProfile
    candidates: list
    seconds: float = 0.0
    llm_summary: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    ai_summary: dict = field(default_factory=dict)

    @property
    def ranked(self):
        return [c for c in self.candidates if c.status == "scored"]

    @property
    def rejected(self):
        return [c for c in self.candidates if c.status != "scored"]


def prepare_jd(jd_text: str) -> JDProfile:
    jd_text = normalize_text(jd_text or "")
    if len(jd_text) < 30:
        raise InputError("The job description is empty or too short to evaluate against.")
    if len(jd_text) > MAX_JD_CHARS:
        raise InputError(f"The job description is too long ({len(jd_text):,} characters; "
                         f"limit {MAX_JD_CHARS:,}).")
    return build_profile(jd_text)


def _reject(c: Candidate, reason: str, gate: bool = False) -> Candidate:
    """gate: rejected by an eligibility rule (degree, experience) rather than an unreadable file."""
    c.status = "rejected"
    c.reject_reason = reason
    c.gate_rejected = gate
    c.category = "❌ Rejected"
    c.score = 0.0
    return c


_GENERIC_DEV_TITLE_RE = re.compile(
    r"\b(?:software|full[\s-]?stack|web|front[\s-]?end|back[\s-]?end|application|it)\s+(?:developer|engineer|"
    r"programmer)\b|\bprogrammer\b", re.IGNORECASE)
_SPECIALIST_TITLE_RE = re.compile(
    r"\b(?:wordpress|shopify|mobile|android|ios|flutter|game|unity|embedded|sap|salesforce|seo|graphic|"
    r"ui/ux|qa|test)\b", re.IGNORECASE)

_SPECIALIST_JD_TOKENS = {
    "ai", "ml", "machine", "learning", "data", "mobile", "android", "ios", "flutter", "devops", "mlops",
    "qa", "sqa", "test", "testing", "security", "cyber", "cloud", "network", "embedded", "game", "bi",
    "analytics", "intelligence", "scientist", "llm", "nlp", "vision", "blockchain", "salesforce", "sap",
    "wordpress", "shopify", "seo", "ui", "ux", "graphic", "database", "dba", "infrastructure", "sre",
}

_UMBRELLA_SKILLS = {"artificial intelligence", "machine learning", "git", "github", "rest api", "json",
                    "agile", "linux", "sql", "html", "css", "excel", "data analysis", "generative ai",
                    "api development", "version control", "microsoft office", "ms office",
                    "performance optimization", "database", "open source", "oop", "scrum", "jira",
                    "system design", "software architecture",
                    "mvc", "mvvm", "mvp", "clean architecture", "orm", "unit testing"}


def _role_relevance(role, profile: JDProfile) -> tuple:
    """Is this job relevant to the JD? (skills used in it, or a similar title)."""
    if not profile.requirements and not profile.title_tokens:
        return True, "JD gives nothing to judge relevance — counted"
    block_skills = extract_skills(role.text)
    idx = CVIndex(role.text, block_skills)
    weight = {}
    for s, m in match_requirements(profile.skills, idx).items():
        if not m.matched:
            continue
        if m.match_type == "direct" or (m.match_type == "inferred" and m.confidence == "strong"):
            weight[s] = 1.0
        elif m.match_type == "inferred" and m.confidence == "moderate":
            weight[s] = 0.5
    hits = set(weight)
    if len(profile.requirements) <= 2 and not profile.title_tokens:
        return (True, "uses " + ", ".join(sorted(hits)[:4])) if hits else (False, "no JD skills in this role")
    points, umbrella, shown = 0.0, 0.0, []
    for req in profile.requirements:
        got = [s for s in req.skills if s in hits]
        if not got:
            continue
        if all(s in _UMBRELLA_SKILLS for s in got):
            umbrella = 0.5
            continue
        core = req.importance == "must" or any(s in profile.title_skills for s in req.skills)
        points += (2.0 if core else 1.0) * max(weight[s] for s in got if s not in _UMBRELLA_SKILLS)
        shown.extend(s for s in got if s not in _UMBRELLA_SKILLS)
    needed = 1.0 if len(profile.requirements) <= 4 else 2.0
    if points + umbrella >= needed:
        return True, "uses " + ", ".join(sorted(set(shown))[:4])
    jd_domain = {t for t in profile.title_tokens if t not in ROLE_FAMILIES}
    role_titles = extract_job_titles(role.header) or [role.header]
    sim, best = scoring.best_title_match(profile, role_titles)
    role_tokens = set(title_tokens(best or role.header))
    if sim >= 0.5 and (not jd_domain or jd_domain & role_tokens):
        return True, "similar job title"
    jd_head = profile.title_tokens[-1] if profile.title_tokens else ""
    if ROLE_FAMILIES.get(jd_head) == "build" and not _SPECIALIST_JD_TOKENS & set(profile.title_tokens) \
            and _GENERIC_DEV_TITLE_RE.search(role.header) and not _SPECIALIST_TITLE_RE.search(role.header):
        return True, "general software-development role"
    return False, "no JD skills or similar title in this role"


def analyse_document(c: Candidate, profile: JDProfile) -> Candidate:
    """Deterministic analysis of one extracted CV (c.text set)."""
    text = c.text
    if detect_prompt_injection(text):
        c.flags.append("⚠️ Contains instructions aimed at AI screening")
        c.warnings.append("The CV contains text that tries to instruct an AI screener. It was scored by "
                          "rules only; review manually.")

    c.degrees = extract_degrees(text)
    c.degrees_display = format_degrees(c.degrees)
    completed = highest_completed_level(c.degrees)
    any_degree = bool(c.degrees) or has_bachelors(text)
    c.still_enrolled = bool(c.degrees) and completed == 0
    if c.degrees and any(d.year is None and not d.in_progress for d in c.degrees):
        c.warnings.append("Degree completion date not evidenced — recruiter verification recommended")
    c.cv_fields = extract_education(text) if any_degree else set()

    exp = analyse_experience(text)
    c.total_years = exp.total_years
    for role in exp.roles:
        role.relevant, role.relevance_reason = _role_relevance(role, profile)
    c.roles = exp.roles
    if exp.roles:
        c.relevant_years = merged_years([r for r in exp.roles if r.relevant])
    else:
        c.relevant_years = exp.total_years
        if exp.total_years:
            c.warnings.append("Experience taken from a stated figure (no dated roles found).")
    for text_range, why in exp.excluded:
        c.notes.append(f"Not counted as employment: {text_range} ({why})")
    c.undated_roles = exp.undated
    for u in exp.undated:
        u.relevant, u.relevance_reason = _role_relevance(u, profile)
        c.notes.append(f"Not counted as experience (no dates given): {u.header}"
                       + (" — looks relevant to this JD" if u.relevant else ""))
    if any(u.relevant for u in exp.undated):
        c.flags.append("🗓️ Relevant job without dates (not counted) — verify")
    c.titles = extract_job_titles(text)

    if profile.bachelors_required and not any_degree:
        return _reject(c, "No bachelor's (or higher) degree found — the JD requires one", gate=True)
    if profile.bachelors_required and c.degrees and completed == 0 and c.still_enrolled:
        c.flags.append("Human verification required: bachelor's degree incomplete")
        return _reject(c, "Only in-progress degrees found — completed bachelor's qualification not evidenced", gate=True)
    if profile.graduates_only and c.still_enrolled:
        return _reject(c, "Degree still in progress — the JD requires graduates only", gate=True)

    cv_skills = extract_skills(text)
    c.matches = match_requirements(profile.skills, CVIndex(text, cv_skills))
    _score(c, profile)
    return c


def _years_while_studying(c: Candidate) -> float:
    """Relevant experience dated before the bachelor's completion (or still studying)."""
    bach = [d for d in c.degrees if d.level == "bachelors" and (d.year or d.in_progress)]
    if not bach or not c.roles:
        return 0.0
    d = max(bach, key=lambda x: x.year or 9999)
    grad = (date.today().year * 12 + date.today().month) if d.in_progress or not d.year else d.year * 12 + 5
    months = 0
    for r in c.roles:
        if r.relevant and r.start < grad:
            months += min(r.end, grad) - r.start
    return round(max(0, months) / 12, 1)


def below_min_experience_cap(relevant: float, required: float) -> float:
    """Score ceiling for a candidate below the JD's stated minimum: 54% for a
    small shortfall, falling to 25% with none of the required experience.
    Short minimums are measured against BELOW_MIN_EXPERIENCE_SCALE_YEARS, so
    0 of "6 months" is a small shortfall and 0 of "7+ years" is a large one."""
    gap = max(0.0, required - max(0.0, relevant)) / max(required, BELOW_MIN_EXPERIENCE_SCALE_YEARS)
    return BELOW_MIN_EXPERIENCE_CAP - (BELOW_MIN_EXPERIENCE_CAP - BELOW_MIN_EXPERIENCE_FLOOR_CAP) * min(1.0, gap)


def _score(c: Candidate, profile: JDProfile):
    c.requirement_status = []
    for req in profile.requirements:
        hits = [c.matches[s] for s in req.skills if s in c.matches and c.matches[s].matched]
        best = max(hits, key=lambda m: m.quality) if hits else None
        c.requirement_status.append((req, scoring.requirement_quality(req, c.matches), best, hits))

    exp_comp, over = scoring.experience_component(c.relevant_years, c.total_years,
                                                  profile.effective_required_years, profile.fresher_role)
    components = [
        scoring.skills_component(profile.requirements, c.matches),
        exp_comp,
        scoring.education_component(profile.education_fields, c.cv_fields),
        scoring.title_component(profile, c.titles),
    ]
    coverage = scoring.must_have_coverage(profile.requirements, c.matches)
    c.score, notes = scoring.combine(components, coverage)
    c.raw_score = scoring.combine_raw(components)
    c.components = {k.name: k for k in components}

    required = profile.required_years
    short = required > 0 and c.relevant_years + MIN_EXPERIENCE_TOLERANCE_YEARS < required
    below_min = short and c.relevant_years < required * MIN_EXPERIENCE_TOLERANCE_RATIO
    shortfall = f"{c.relevant_years:g} of {required:g} required years of relevant experience"
    if below_min:
        if BELOW_MIN_EXPERIENCE_POLICY == "reject":
            c.notes = [n for n in c.notes if not n.startswith(("Not scored", "Capped"))] + notes
            _reject(c, f"Below the JD's minimum experience: {shortfall}"
                       + (f" (total {c.total_years:g} yrs, the rest not relevant to this role)"
                          if c.total_years > c.relevant_years else ""), gate=True)
            return
        cap = below_min_experience_cap(c.relevant_years, required)
        capped = scoring.soft_cap(c.score, cap)
        if capped < c.score:
            notes.append(f"Capped at {capped:.0f}% (uncapped {c.score:.0f}%) — below the JD's "
                         f"minimum experience: {shortfall}")
            c.score = round(capped, 2)
    missing_title = sorted(sk for sk in profile.title_skills
                           if not (c.matches.get(sk) and c.matches[sk].matched))
    if missing_title:
        capped = scoring.soft_cap(c.score, MISSING_TITLE_SKILL_CAP)
        if capped < c.score:
            notes.append(f"Capped at {capped:.0f}% (uncapped {c.score:.0f}%) — no evidence of "
                         f"{', '.join(missing_title)}, named in the job title")
            c.score = round(capped, 2)
    c.category = scoring.classify_candidate(c.score)
    c.notes = [n for n in c.notes if not n.startswith(("Not scored", "Capped"))] + notes

    c.flags = [f for f in c.flags if f not in ("💼 Over-qualified", "🎓 New graduate")
               and not f.startswith("⏳")]
    if below_min:
        c.flags.append(f"⏳ Below minimum experience ({c.relevant_years:g} of {required:g} yrs)")
    elif short:
        c.flags.append(f"⏳ Just under minimum experience ({c.relevant_years:g} of {required:g} yrs) — "
                       f"accepted as meeting it")
    c.flags = [f for f in c.flags if not f.startswith("📚 ")]
    study = _years_while_studying(c)
    if study >= 0.5:
        c.flags.append(f"📚 {study:g} of {c.relevant_years:g} relevant yrs were before graduation — "
                       f"verify it was professional work")
    if over and (profile.effective_required_years > 0 or profile.prefers_freshers):
        c.flags.append("💼 Over-qualified")
    years_done = [d.year for d in c.degrees if d.year and not d.in_progress]
    if years_done and date.today().year - max(years_done) <= 2 and c.total_years < 1.0:
        c.flags.append("🎓 New graduate")


def missing_concepts(c: Candidate) -> list:
    """Unmatched CONCEPT requirements (the only kind worth asking the LLM), must-haves first."""
    rank = {"must": 0, "normal": 1, "nice": 2}
    return sorted((r for r, q, *_ in c.requirement_status if q <= 0 and is_concept_requirement(r)),
                  key=lambda r: rank[r.importance])


def llm_plan(cands: list, opts: LLMOptions, only_uids=None):
    """[(candidate, requirements_to_check)] the LLM would process, plus #skipped for injection."""
    todo, skipped = [], 0
    for c in cands:
        if c.status != "scored":
            continue
        missing = missing_concepts(c)
        if not missing:
            continue
        if only_uids is not None:
            if c.uid not in only_uids:
                continue
        elif c.score < opts.min_score:
            continue
        if any(f.startswith("⚠️ Contains instructions") for f in c.flags):
            skipped += 1
            continue
        todo.append((c, missing))
    todo.sort(key=lambda t: (-t[0].score, t[0].name))
    if only_uids is None:
        todo = todo[:max(0, opts.max_candidates)]
    return todo, skipped


def _llm_pass(cands: list, profile: JDProfile, opts: LLMOptions, progress=None, only_uids=None) -> dict:
    """
    progress: optional callable(done, total, message).
    only_uids: verify exactly these candidates (ignores top-N / min-score).
    """
    summary = {"enabled": True, "model": opts.model, "attempted": 0, "succeeded": 0, "failed": 0,
               "accepted_matches": 0, "rejected_claims": 0, "skipped_budget": 0, "skipped_injection": 0,
               "seconds": 0.0, "errors": []}
    try:
        if opts.provider == "cloud":
            if not opts.cloud_consent:
                summary["errors"].append("Cloud CV processing requires explicit consent")
                summary["failed"] += 1
                return summary
            client = CloudClient(opts.model, opts.api_key or get_cloud_api_key(), opts.base_url)
        else:
            client = OllamaClient(opts.model)
    except (ValueError, LLMError) as exc:
        summary["errors"].append(str(exc))
        summary["failed"] += 1
        return summary
    parallel = LLM_CLOUD_PARALLEL if opts.provider == "cloud" else LLM_PARALLEL

    todo, skipped_injection = llm_plan(cands, opts, only_uids)
    summary["skipped_injection"] = skipped_injection
    if only_uids is None:
        eligible = len(llm_plan(cands, LLMOptions(True, opts.model, max_candidates=10 ** 6,
                                                  min_score=opts.min_score))[0])
        summary["skipped_not_top"] = max(0, eligible - len(todo))
    summary["planned"] = len(todo)

    start = time.time()
    consecutive_failures = 0
    stop_reason = ""

    def work(item):
        c, missing = item
        labels = {r.label.replace(" / ", " or "): r for r in missing}
        return c, labels, client.verify_requirements(list(labels), c.text)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        i = 0
        while i < len(todo):
            if stop_reason:
                summary["skipped_budget"] += len(todo) - i
                break
            if time.time() - start >= opts.budget_seconds:
                stop_reason = "time budget reached"
                continue
            chunk = todo[i:i + max(1, parallel)]
            if progress:
                progress(i, len(todo), "Checking " + ", ".join(c.name for c, _ in chunk))
            i += len(chunk)
            futures = [pool.submit(work, it) for it in chunk]
            for fut, (c, _) in zip(futures, chunk):
                summary["attempted"] += 1
                try:
                    c, labels, res = fut.result()
                except LLMUnavailable as exc:
                    summary["failed"] += 1
                    summary["errors"].append(str(exc))
                    stop_reason = "LLM unavailable"
                    continue
                except LLMError as exc:
                    summary["failed"] += 1
                    summary["errors"].append(f"{c.name}: {exc}")
                    consecutive_failures += 1
                    if consecutive_failures >= LLM_CIRCUIT_BREAKER_FAILURES:
                        stop_reason = f"{consecutive_failures} consecutive LLM failures"
                    continue
                except Exception as exc:
                    logger.exception("unexpected LLM error")
                    summary["failed"] += 1
                    summary["errors"].append(f"{c.name}: unexpected {type(exc).__name__}")
                    continue
                consecutive_failures = 0
                summary["succeeded"] += 1
                c.llm_used = True
                for v in res.verdicts:
                    req = labels.get(v.requirement)
                    if req is None:
                        continue
                    if v.accepted and not evidence_supports(req.skills, v.evidence):
                        summary["rejected_claims"] += 1
                        c.llm_rejections.append(f"{req.label}: the CV line the model pointed to "
                                                f"('{v.evidence[:60]}') says nothing related")
                        continue
                    if v.accepted:
                        skill = req.skills[0]
                        c.matches[skill] = SkillMatch(skill, True, "llm", f"LLM ({res.model})",
                                                      v.evidence, MATCH_CREDIT["llm"])
                        summary["accepted_matches"] += 1
                    elif v.satisfied:
                        summary["rejected_claims"] += 1
                        c.llm_rejections.append(f"{req.label}: {v.rejection}")
                _score(c, profile)
                if progress:
                    progress(i, len(todo), f"Checked {c.name} ({res.seconds:.0f}s)")
    summary["seconds"] = round(time.time() - start, 1)
    summary["stop_reason"] = stop_reason
    return summary


def evaluate_batch(jd_text: str, files: list, llm: LLMOptions = None, progress=None) -> BatchResult:
    """
    files: [(filename, bytes)]. progress: optional callable(done, total, message).
    Raises InputError for invalid batch input (bad JD, no files, too many files).
    """
    t0 = time.time()
    profile = prepare_jd(jd_text)
    if not files:
        raise InputError("Upload at least one CV.")
    if len(files) > MAX_FILES_PER_BATCH:
        raise InputError(f"Too many files ({len(files)}); the limit is {MAX_FILES_PER_BATCH} per batch.")

    candidates = []
    first_by_hash = {}
    for i, (name, data) in enumerate(files):
        c = Candidate(uid=f"{i:04d}-{name}", name=name or f"file-{i}")
        c.sha256 = hashlib.sha256(data or b"").hexdigest()
        candidates.append(c)

    empty_hash = hashlib.sha256(b"").hexdigest()

    def dedupe_key(c):
        return c.uid if c.sha256 == empty_hash else c.sha256

    def work(item):
        c, data = item
        if first_by_hash[dedupe_key(c)] is not c:
            return c
        try:
            res = extract_document(data, detect_type(c.name))
        except ExtractionError as exc:
            return _reject(c, f"Unreadable file: {exc}")
        c.file_type = res.file_type
        c.warnings.extend(res.warnings)
        if res.ocr_pages:
            c.flags.append("🔍 OCR used")
        c.text = res.text
        if len(c.text) > MAX_CV_TEXT_CHARS:
            c.text = c.text[:MAX_CV_TEXT_CHARS]
            c.warnings.append(f"Very long document: only the first {MAX_CV_TEXT_CHARS:,} characters were analysed.")
        if len(res.text.strip()) < MIN_TEXT_CHARS:
            return _reject(c, "No readable text (scanned image without OCR, or empty document)")
        try:
            return analyse_document(c, profile)
        except Exception as exc:
            logger.exception("analysis failed for one CV")
            return _reject(c, f"Internal error while analysing this CV ({type(exc).__name__})")

    for c in candidates:
        first_by_hash.setdefault(dedupe_key(c), c)

    done = 0
    total = len(candidates)
    with ThreadPoolExecutor(max_workers=max(1, min(EXTRACTION_WORKERS, total))) as pool:
        for c in pool.map(work, zip(candidates, [d for _, d in files])):
            done += 1
            if progress:
                progress(done, total, f"Analysed {c.name}")

    for c in candidates:
        orig = first_by_hash[dedupe_key(c)]
        if orig is not c:
            c.status = "duplicate"
            c.duplicate_of = orig.name
            c.reject_reason = f"Duplicate of '{orig.name}' (identical file)"
            c.category = "⚪ Duplicate"

    warnings = list(profile.warnings)
    llm_summary = {"enabled": False}
    if llm and llm.enabled:
        if not llm.model:
            warnings.append("LLM mode was on but no model was selected — rules only.")
        else:
            llm_summary = _llm_pass(candidates, profile, llm, progress)
            if llm_summary.get("failed"):
                warnings.append(f"LLM failed for {llm_summary['failed']} CV(s); those use rule-based "
                                f"matching only.")
            if llm_summary.get("skipped_budget"):
                warnings.append(f"LLM skipped {llm_summary['skipped_budget']} CV(s) "
                                f"({llm_summary.get('stop_reason')}).")

    return BatchResult(profile, _rank(candidates), round(time.time() - t0, 1), llm_summary, warnings)


def _rank(candidates: list) -> list:
    """Score desc, then uncapped score, then relevant experience, then name — never processing order."""
    ranked = sorted((c for c in candidates if c.status == "scored"),
                    key=lambda c: (-c.score, -c.raw_score, -c.relevant_years, c.name.lower(), c.uid))
    for i, c in enumerate(ranked, 1):
        c.rank = i
    return ranked + [c for c in candidates if c.status != "scored"]


def refine_with_llm(result: BatchResult, opts: LLMOptions, only_uids=None, progress=None) -> BatchResult:
    """
    Second, optional step: ask the local LLM about requirements the rules
    could not match — for the top `opts.max_candidates` candidates, or
    exactly `only_uids`. Updates scores in place and re-ranks. Rule-based
    results are always available first; this never blocks them.
    """
    summary = _llm_pass(result.candidates, result.profile, opts, progress, only_uids)
    prev = result.llm_summary if result.llm_summary.get("enabled") else {}
    for k in ("attempted", "succeeded", "failed", "accepted_matches", "rejected_claims", "seconds"):
        summary[k] = round(summary.get(k, 0) + prev.get(k, 0), 1)
    summary["errors"] = prev.get("errors", []) + summary.get("errors", [])
    result.llm_summary = summary
    result.candidates = _rank(result.candidates)
    return result


def component_weights_note(profile: JDProfile) -> str:
    """Human-readable effective weights for this JD."""
    from config import EXP_NO_MINIMUM_WEIGHT
    has_min = profile.effective_required_years > 0
    avail = {"skills": bool(profile.requirements), "experience": has_min or not profile.fresher_role,
             "education": bool(profile.education_fields), "title": bool(profile.title_tokens)}
    w = dict(WEIGHTS, experience=WEIGHTS["experience"] * (1.0 if has_min else EXP_NO_MINIMUM_WEIGHT))
    total = sum(w[k] for k, v in avail.items() if v) or 1
    parts = [f"{k} {w[k] / total * 100:.0f}%" for k, v in avail.items() if v]
    return ", ".join(parts) + " (title also needs job titles in the CV)"


__all__ = ["evaluate_batch", "refine_with_llm", "llm_plan", "missing_concepts", "prepare_jd", "LLMOptions", "Candidate", "BatchResult", "InputError",
           "component_weights_note", "OVERQUALIFIED_MARGIN_YEARS", "title_tokens"]

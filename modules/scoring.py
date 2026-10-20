"""
The scoring model.

Rules:
  * Every component reports whether the JD gave it something to
    measure. Weights are renormalised across AVAILABLE components only,
    so candidates are never marked down for something the JD never asked.
  * Skills are scored per REQUIREMENT (an alternative group such as
    "Git / SVN" is one requirement satisfied by either), weighted by
    importance (must / normal / nice) and by how the match was made.
  * Experience is a MINIMUM. Meeting or exceeding the JD's minimum earns
    full credit; being over-qualified is reported but never lowers the
    score. Only RELEVANT experience counts towards the minimum.
  * Post-combination gates (must-haves, stated minimum experience, a
    skill named in the title) cap the score. Caps are SOFT (soft_cap):
    a flat cap gave every capped candidate the identical score, so a
    5-year candidate and a fresh graduate both read "54.0".
"""
from dataclasses import dataclass

from config import (
    EXP_AT_MINIMUM, EXP_DEPTH_YEARS, EXP_NO_MINIMUM_BASE, EXP_NO_MINIMUM_WEIGHT, EXP_SHORTFALL_EXPONENT, MUST_HAVE_LOW_COVERAGE, MUST_HAVE_MIN_COVERAGE, MUST_HAVE_SCORE_CAP,
    MUST_HAVE_SCORE_FLOOR_CAP,
    MUST_HAVE_WEIGHT, NICE_TO_HAVE_WEIGHT, OVERQUALIFIED_MARGIN_YEARS, SOFT_CAP_SPREAD, THRESHOLDS, WEIGHTS,
)
from modules.education import get_education_score
from modules.jd_profile import ROLE_FAMILIES, title_tokens


@dataclass
class Component:
    name: str
    value: float
    available: bool
    detail: str = ""
    weight: float = 1.0


_IMPORTANCE_WEIGHT = {"must": MUST_HAVE_WEIGHT, "normal": 1.0, "nice": NICE_TO_HAVE_WEIGHT}


def requirement_quality(req, matches: dict) -> float:
    """
    Quality of one requirement: each item scores its best alternative;
    the best `min_hits` items are averaged (an examples list needs two
    examples for full credit, an alternative set needs one).
    """
    items = getattr(req, "items", None) or (tuple(req.skills),)
    item_q = sorted((max((matches[s].quality for s in it if s in matches), default=0.0) for it in items),
                    reverse=True)
    k = max(1, min(getattr(req, "min_hits", 1), len(item_q)))
    return sum(item_q[:k]) / k


def requirement_weights(requirements) -> list:
    """Importance weight of each requirement, shared within its JD line.

    One sentence naming several tools ("hands-on experience with LangChain,
    RAG, LlamaIndex and vector databases") is one area of the job, not four
    separate ones, and "backend API development in FastAPI/Flask" must not
    count twice. A line with k requirements weighs (k + 1) / 2 times a
    single one in total: more than one requirement, less than k."""
    per_line = {}
    for r in requirements:
        if r.source:
            per_line[r.source] = per_line.get(r.source, 0) + 1
    out = []
    for r in requirements:
        k = per_line.get(r.source, 1) if r.source else 1
        out.append(_IMPORTANCE_WEIGHT.get(r.importance, 1.0) * (k + 1) / (2 * k))
    return out


def skills_component(requirements, matches: dict) -> Component:
    if not requirements:
        return Component("skills", 0.0, False, "JD listed no recognisable skills")
    earned = possible = 0.0
    hit = 0
    for req, w in zip(requirements, requirement_weights(requirements)):
        q = requirement_quality(req, matches)
        possible += w
        earned += w * q
        hit += q > 0
    return Component("skills", earned / possible if possible else 0.0, True,
                     f"{hit}/{len(requirements)} JD requirements met")


def experience_component(relevant_years: float, total_years: float, required_years: float,
                         fresher_role: bool = False):
    """
    Returns (Component, overqualified_flag).

    More RELEVANT experience is better, with diminishing returns:
      * JD minimum met → EXP_AT_MINIMUM, rising to 1.0 at EXP_DEPTH_YEARS
        beyond it (so 6 relevant years beat 2 on a "2+ years" JD);
        below it, EXP_AT_MINIMUM × (relevant / minimum) ** EXP_SHORTFALL_EXPONENT.
      * No minimum stated → scored at EXP_NO_MINIMUM_WEIGHT weight, from
        EXP_NO_MINIMUM_BASE (none) to 1.0 (EXP_DEPTH_YEARS relevant years).
      * Fresher / trainee JD with no minimum → not scored: extra years must
        not push fresh graduates down for a role meant for them.
    Over-qualification is a flag only; it never lowers the score.
    """
    rel = max(0.0, relevant_years)
    over = rel >= max(required_years, 0) + OVERQUALIFIED_MARGIN_YEARS
    base = f"{relevant_years:g} yrs relevant (total {total_years:g})"
    if required_years <= 0:
        if fresher_role:
            return Component("experience", 0.0, False,
                             f"{base} — entry-level JD with no minimum: not scored"), over
        value = EXP_NO_MINIMUM_BASE + (1.0 - EXP_NO_MINIMUM_BASE) * min(1.0, rel / EXP_DEPTH_YEARS)
        return Component("experience", value, True,
                         f"{base} — JD sets no minimum; more relevant experience scores higher "
                         f"(half weight)", EXP_NO_MINIMUM_WEIGHT), over
    if rel >= required_years:
        value = EXP_AT_MINIMUM + (1.0 - EXP_AT_MINIMUM) * min(1.0, (rel - required_years) / EXP_DEPTH_YEARS)
        return Component("experience", value, True,
                         f"{base} ✅ meets minimum {required_years:g}"
                         + (f" (+{rel - required_years:g} yrs beyond it)" if rel > required_years else "")), over
    ratio = rel / required_years
    value = EXP_AT_MINIMUM * ratio ** EXP_SHORTFALL_EXPONENT
    return Component("experience", value, True,
                     f"{base} — below minimum {required_years:g} {'🟡' if ratio >= 0.5 else '❌'}"), over


def education_component(jd_fields, cv_fields) -> Component:
    """Field-of-study alignment, averaged over the JD's fields."""
    if not jd_fields:
        return Component("education", 0.0, False, "JD stated no field of study")
    if not cv_fields:
        return Component("education", 0.0, True, "No field of study detected in CV")
    best_names = []
    for jd_field in jd_fields:
        best, best_name = 0, None
        for cv_field in cv_fields:
            s = get_education_score(jd_field, cv_field)
            if s > best:
                best, best_name = s, cv_field
        if best_name:
            best_names.append(best_name)
    value = max(get_education_score(j, c) for j in jd_fields for c in cv_fields) / 3.0
    detail = ", ".join(list(dict.fromkeys(best_names))[:3]) if best_names else "No field match"
    return Component("education", min(1.0, value), True, detail)


_DOMAIN_OF = {
    "flutter": "mobile", "android": "mobile", "ios": "mobile", "kotlin": "mobile", "swift": "mobile",
    "app": "mobile", "apps": "mobile", "native": "mobile",
    "ml": "ai", "machine": "ai", "learning": "ai", "deep": "ai", "nlp": "ai", "llm": "ai", "genai": "ai",
    "vision": "ai", "artificial": "ai",
    "stack": "web", "fullstack": "web", "mern": "web", "mean": "web", "frontend": "web", "front": "web",
    "end": "web", "front-end": "web", "react": "web", "angular": "web", "vue": "web", "wordpress": "web",
    "human": "hr", "resources": "hr", "resource": "hr", "people": "hr", "talent": "hr", "recruitment": "hr",
}
_NO_DOMAIN = {"software", "application", "applications", "it", "technical", "technology", "computer", "general"}


def _domains(tokens) -> set:
    return {_DOMAIN_OF.get(t, t) for t in tokens if t not in ROLE_FAMILIES and t not in _NO_DOMAIN}


def _title_similarity(jd_tokens, cv_tokens) -> float:
    if not jd_tokens or not cv_tokens:
        return 0.0
    jd_dom, cv_dom = _domains(jd_tokens), _domains(cv_tokens)
    mismatch = 0.5 if jd_dom and cv_dom and not jd_dom & cv_dom else 1.0
    jd_head, cv_head = jd_tokens[-1], cv_tokens[-1]
    if jd_head == cv_head:
        head = 1.0
    elif ROLE_FAMILIES.get(jd_head) and ROLE_FAMILIES.get(jd_head) == ROLE_FAMILIES.get(cv_head):
        head = 0.75
    elif {ROLE_FAMILIES.get(jd_head), ROLE_FAMILIES.get(cv_head)} == {"build", "analyse"}:
        head = 0.4
    else:
        head = 0.0
    if jd_dom & cv_dom and ROLE_FAMILIES.get(cv_head):
        head = max(head, 0.9 if head >= 0.75 else 0.5)
    a, b = set(jd_tokens), set(cv_tokens)
    overlap = len(a & b) / min(len(a), len(b))
    return (0.6 * head + 0.4 * overlap) * mismatch


def best_title_match(profile, cv_titles):
    best, best_title = 0.0, ""
    for t in cv_titles or []:
        s = _title_similarity(profile.title_tokens, title_tokens(t))
        if s > best:
            best, best_title = s, t
    return best, best_title


def title_component(profile, cv_titles) -> Component:
    if not profile.title_tokens:
        return Component("title", 0.0, False, "No job title found in JD")
    if not cv_titles:
        return Component("title", 0.0, True, "No job title or headline found in CV")
    best, best_title = best_title_match(profile, cv_titles)
    detail = f"{best_title} ≈ {profile.title}" if best_title else "No role match"
    return Component("title", best, True, detail)


def soft_cap(raw: float, cap: float) -> float:
    """Ceiling that keeps candidates apart: min(raw, cap × (1 − s + s × raw/100)).
    Monotone in raw, never above cap, never raises a score."""
    return min(raw, cap * (1.0 - SOFT_CAP_SPREAD + SOFT_CAP_SPREAD * max(0.0, min(raw, 100.0)) / 100.0))


def must_have_cap(coverage: float) -> float:
    """Ceiling for a candidate meeting too few must-haves: lower the fewer they meet.
    30% at none met, 45% at MUST_HAVE_LOW_COVERAGE, then rising continuously
    (slowly at first: missing half the must-haves stays near the top of
    Moderate) to no ceiling (100) at MUST_HAVE_MIN_COVERAGE."""
    c = max(0.0, coverage)
    if c <= MUST_HAVE_LOW_COVERAGE:
        return MUST_HAVE_SCORE_FLOOR_CAP + (MUST_HAVE_SCORE_CAP - MUST_HAVE_SCORE_FLOOR_CAP) * c / MUST_HAVE_LOW_COVERAGE
    share = min(1.0, (c - MUST_HAVE_LOW_COVERAGE) / (MUST_HAVE_MIN_COVERAGE - MUST_HAVE_LOW_COVERAGE))
    return MUST_HAVE_SCORE_CAP + (100.0 - MUST_HAVE_SCORE_CAP) * share ** 2


def must_have_coverage(requirements, matches: dict):
    """Weighted share of the must-haves met (weights shared per JD line, as in skills)."""
    musts = [(r, w) for r, w in zip(requirements, requirement_weights(requirements)) if r.importance == "must"]
    if not musts:
        return None
    return sum(requirement_quality(r, matches) * w for r, w in musts) / sum(w for _, w in musts)


def combine(components, must_coverage=None):
    """Renormalise over available components, apply the must-have cap.
    Returns (final_pct, notes); the uncapped value is in the notes and is
    also exposed via combine_raw() for tie-breaking."""
    available = [c for c in components if c.available]
    notes = []
    skipped = [c.name for c in components if not c.available]
    if skipped:
        notes.append("Not scored (JD silent / nothing to measure): " + ", ".join(skipped))
    if not available:
        return 0.0, notes + ["No scoreable signal in this JD"]
    total_weight = sum(WEIGHTS[c.name] * c.weight for c in available)
    combined = sum(WEIGHTS[c.name] * c.weight * c.value for c in available) / total_weight * 100.0

    if must_coverage is not None and must_coverage < MUST_HAVE_MIN_COVERAGE:
        capped = soft_cap(combined, must_have_cap(must_coverage))
        if capped < combined:
            notes.append(f"Capped at {capped:.0f}% (uncapped {combined:.0f}%) — only "
                         f"{must_coverage * 100:.0f}% of must-have requirements met")
            combined = capped
    return round(min(100.0, max(0.0, combined)), 2), notes


def combine_raw(components) -> float:
    """Weighted score before the must-have cap (used only to order ties)."""
    available = [c for c in components if c.available]
    if not available:
        return 0.0
    total_weight = sum(WEIGHTS[c.name] * c.weight for c in available)
    return round(sum(WEIGHTS[c.name] * c.weight * c.value for c in available) / total_weight * 100.0, 2)


def classify_candidate(combined_pct):
    if combined_pct >= THRESHOLDS["Highly Suitable"]:
        return "✅ Highly Suitable"
    if combined_pct >= THRESHOLDS["Suitable"]:
        return "🟢 Suitable"
    if combined_pct >= THRESHOLDS["Moderate"]:
        return "🟡 Moderate"
    return "❌ Not Suitable"

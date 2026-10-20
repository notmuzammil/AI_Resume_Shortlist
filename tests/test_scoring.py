import pytest

from config import MUST_HAVE_SCORE_CAP
from modules import scoring
from modules.jd_profile import Requirement
from modules.matcher import SkillMatch


def m(skill, q=1.0):
    return SkillMatch(skill, q > 0, "direct" if q else "none", "", "", q)


@pytest.mark.parametrize("relevant,required,expected", [
    (2.0, 2.0, 0.85),
    (3.5, 2.0, 0.925),
    (5.0, 2.0, 1.0),
    (10.0, 2.0, 1.0),
    (25.0, 2.0, 1.0),
    (1.0, 2.0, 0.85 * 0.5 ** 0.85),
    (0.0, 2.0, 0.0),
])
def test_experience_credit(relevant, required, expected):
    comp, _ = scoring.experience_component(relevant, relevant, required)
    assert comp.available and comp.value == pytest.approx(expected)


def test_overqualified_is_a_flag_only():
    comp, over = scoring.experience_component(12.0, 12.0, 2.0)
    assert over and comp.value == 1.0


def test_no_minimum_still_prefers_relevant_experience_at_half_weight():
    none, _ = scoring.experience_component(0.0, 0.0, 0.0)
    some, _ = scoring.experience_component(1.5, 1.5, 0.0)
    lots, _ = scoring.experience_component(5.0, 5.0, 0.0)
    assert none.available and none.weight == 0.5 and none.value == 0.5
    assert none.value < some.value < lots.value == 1.0


def test_fresher_jd_without_minimum_does_not_score_experience():
    comp, _ = scoring.experience_component(5.0, 5.0, 0.0, fresher_role=True)
    assert not comp.available


def test_alternatives_and_examples_quality():
    alt = Requirement(("git", "svn"))
    assert scoring.requirement_quality(alt, {"svn": m("svn")}) == 1.0
    ex = Requirement(("mongodb", "mysql", "sql server"), items=(("mongodb",), ("mysql",), ("sql server",)),
                     min_hits=2)
    assert scoring.requirement_quality(ex, {"mongodb": m("mongodb")}) == 0.5
    assert scoring.requirement_quality(ex, {"mongodb": m("mongodb"), "mysql": m("mysql")}) == 1.0


def test_importance_weights():
    reqs = [Requirement(("python",), "must"), Requirement(("docker",), "nice")]
    hi = scoring.skills_component(reqs, {"python": m("python")}).value
    lo = scoring.skills_component(reqs, {"docker": m("docker")}).value
    assert hi > lo


def test_must_have_cap_and_raw_score():
    comps = [scoring.Component("skills", 0.9, True), scoring.Component("title", 1.0, True)]
    score, notes = scoring.combine(comps, must_coverage=0.2)
    assert score < MUST_HAVE_SCORE_CAP and any("Capped" in n for n in notes)
    assert scoring.combine_raw(comps) > MUST_HAVE_SCORE_CAP
    assert scoring.combine(comps, must_coverage=0.0)[0] < score < scoring.combine(comps, must_coverage=0.45)[0]


def test_soft_cap_keeps_capped_candidates_apart():
    assert scoring.soft_cap(90, 54) > scoring.soft_cap(70, 54) > scoring.soft_cap(60, 54)
    assert scoring.soft_cap(100, 54) == 54 and scoring.soft_cap(30, 54) == 30


def test_renormalisation_over_available_components():
    comps = [scoring.Component("skills", 0.5, True), scoring.Component("experience", 0.0, False),
             scoring.Component("education", 0.0, False), scoring.Component("title", 0.0, False)]
    assert scoring.combine(comps)[0] == 50.0


def test_cv_without_titles_is_not_advantaged():
    from modules.jd_profile import build_profile
    p = build_profile("Position: AI Engineer\nPython")
    none = scoring.title_component(p, [])
    other = scoring.title_component(p, ["Account Manager"])
    assert none.available and none.value == 0.0 and other.value == 0.0


def test_categories():
    assert scoring.classify_candidate(80).startswith("✅")
    assert "Suitable" in scoring.classify_candidate(60)
    assert "Moderate" in scoring.classify_candidate(30)
    assert "Not Suitable" in scoring.classify_candidate(10)


def test_must_have_ceiling_has_no_cliff_at_the_threshold():
    from config import MUST_HAVE_MIN_COVERAGE
    caps = [scoring.must_have_cap(c / 100) for c in range(0, round(MUST_HAVE_MIN_COVERAGE * 100) + 1)]
    assert caps == sorted(caps) and caps[0] == 30 and caps[-1] == 100
    assert max(b - a for a, b in zip(caps, caps[1:])) < 5
    assert scoring.must_have_cap(0.5) < 65


def test_one_jd_line_listing_many_tools_is_one_area_of_the_job():
    from modules.jd_profile import Requirement
    line = "Must have hands-on experience with LangChain, RAG, LlamaIndex and vector databases"
    reqs = [Requirement(("python",), "must", "Strong Python")] + \
           [Requirement((s,), "must", line) for s in ("langchain", "rag", "llamaindex", "vector database")]
    w = scoring.requirement_weights(reqs)
    assert w[0] > w[1] and sum(w[1:]) < 4 * w[0] and sum(w[1:]) > w[0]
    assert scoring.must_have_coverage(reqs, {"python": m("python")}) > 0.25

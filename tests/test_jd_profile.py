import pytest

from modules.extractor import normalize_text
from modules.jd_profile import build_profile, detect_seniority, extract_required_years


def labels(profile):
    return {r.label: r.importance for r in profile.requirements}


@pytest.mark.parametrize("text,years", [
    ("Minimum 6 months of professional mobile app development experience.", 0.5),
    ("3+ years of experience with Python", 3.0),
    ("Looking for Freelance Python + React Developer (3+ Years)", 3.0),
    ("3+ years TypeScript / Node.js", 3.0),
    ("2-4 years of relevant experience", 2.0),
    ("at least five years of industry experience", 5.0),
    ("Bachelor's degree (16 years) in Computer Science", 0.0),
    ("Bachelor's degree (16 years of education)", 0.0),
    ("We have 17+ background services", 0.0),
    ("No experience mentioned here.", 0.0),
    ("With over 20 years of experience, our company serves clients worldwide.", 0.0),
    ("We bring 15+ years of industry experience. You need 2+ years of experience in Go.", 2.0),
    ("The ideal candidate has 4 years of experience with our stack.", 4.0),
])
def test_required_years(text, years):
    assert extract_required_years(text) == years


def test_seniority_from_title_not_body():
    body = "You will lead code reviews and report to the engineering manager."
    assert detect_seniority(body, "Python Developer") == "mid"
    assert detect_seniority(body, "Senior Python Developer") == "senior"
    assert detect_seniority(body, "AI Engineer (Trainee)") == "entry"


def test_alternatives_become_one_requirement(jd_texts):
    p = build_profile(jd_texts["ai"])
    lab = labels(p)
    assert "git / svn" in lab
    assert "django / flask" in lab
    assert "mysql / nosql" in lab
    assert "openai / gemini" in lab
    assert "svn" not in lab and "git" not in lab


def test_title_line_is_not_a_requirement(jd_texts):
    p = build_profile(jd_texts["fullstack"])
    assert "full stack" not in labels(p)


def test_examples_list_and_composite_stack(jd_texts):
    lab = labels(build_profile(jd_texts["fullstack"]))
    assert "mern / mean" in lab
    assert "mongodb, mysql, sql server (any 2)" in lab
    assert "react" not in lab and "angular" not in lab
    assert "git" in lab and "rest api" in lab


def test_team_mentions_and_soft_skills_are_not_requirements(jd_texts):
    lab = labels(build_profile(jd_texts["mobile"]))
    assert "qa" not in lab
    assert "problem solving" not in lab
    assert build_profile(jd_texts["mobile"]).required_years == 0.5


def test_inline_preferred_only_applies_to_bracket():
    jd = ("Python + React Developer\nRequired Skills:\n"
          "• 3+ years of experience with Python (Django/Flask preferred)\n"
          "• Experience with Docker\n• Kubernetes is a plus\n• AWS/OpenStack (Plus)\n")
    lab = labels(build_profile(jd))
    assert lab["python"] == "normal" or lab["python"] == "must"
    assert lab["django / flask"] == "nice"
    assert lab["kubernetes"] == "nice"
    assert lab["aws / openstack"] == "nice"


def test_degree_preferred_is_not_a_gate():
    jd = "MLOps Engineer\nBachelor's or Master's degree in Computer Science preferred.\nPython, Docker."
    assert not build_profile(jd).bachelors_required
    jd2 = "AI Engineer\nBachelor's degree in Computer Science.\nPython."
    assert build_profile(jd2).bachelors_required


def test_curly_apostrophe_and_fancy_unicode_heading():
    jd = normalize_text("JAVA DEVELOPER\n🔹 𝐒𝐤𝐢𝐥𝐥𝐬 𝐑𝐞𝐪𝐮𝐢𝐫𝐞𝐝:\n• Core Java & J2EE\n"
                        "Bachelor’s degree in Computer Science")
    p = build_profile(jd)
    assert p.bachelors_required
    assert "java" in p.skills


def test_empty_jd_profile_has_warning():
    p = build_profile("We are hiring. Apply now!")
    assert not p.requirements and p.warnings


def test_few_requirements_warns():
    p = build_profile("Position: Sales Executive\nMust know Excel.")
    assert 0 < len(p.requirements) < 3
    assert any("recognisable skill" in w for w in p.warnings)


def test_fresher_role_vs_freshers_merely_eligible():
    from modules.jd_profile import build_profile
    assert build_profile("Position: AI Engineer (Trainee)\nPython").fresher_role
    assert build_profile("Position: Python Developer\nFresh graduates are preferred.\nPython").fresher_role
    p = build_profile("Backend Developer\nRequirements\nFresh graduates with good fundamentals are eligible "
                      "to apply\nPython, REST APIs")
    assert p.prefers_freshers and not p.fresher_role

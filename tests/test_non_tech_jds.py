"""HR / non-technical JDs and CVs, JD headings, and batch edge cases."""
import time

from modules.education import extract_degrees, extract_education
from modules.experience import extract_job_titles
from modules.jd_profile import build_profile, extract_job_title
from modules.pipeline import evaluate_batch
from modules.skill_extractor import skill_mentions

HR_JD = """HR Generalist

ABOUT US
We are a fast-growing logistics company.

WHAT YOU'LL DO
- Manage end-to-end recruitment: screening, interviews and onboarding.
- Maintain employee records and HRIS, process payroll inputs.

REQUIREMENTS
- Bachelor's degree in Human Resource Management, Business Administration or related field.
- 2+ years of experience in HR or talent acquisition.
- Experience with HRIS / ATS tools (e.g. BambooHR, Workday).
- Knowledge of payroll, compensation and benefits.
- Proficiency in MS Office (Excel, Word, PowerPoint).

IDEAL CANDIDATE
Someone who loves working with people.
"""

HR_CV = """SARAH AHMED
HR Generalist
sarah@example.com

WORK EXPERIENCE
HR Generalist — Bright Logistics, Karachi
March 2022 – Present
• Ran end-to-end recruitment using BambooHR; payroll inputs, compensation and benefits
HR Coordinator — Zeta Textiles, Lahore
June 2020 – February 2022
• Coordinated interviews; maintained employee records in MS Excel

EDUCATION
BBA (Human Resource Management), University of Karachi, 2020
"""

DEV_CV = """Ali Raza
Full Stack Developer
EXPERIENCE
Full Stack Developer, Acme — Jan 2021 – Present
React, Node.js, MongoDB, Express, Git, REST APIs.
EDUCATION
BS Computer Science, FAST, 2020
"""


def test_non_tech_job_titles_detected():
    assert extract_job_title(HR_JD) == "HR Generalist"
    assert extract_job_title("Talent Acquisition Specialist\nWe hire.") == "Talent Acquisition Specialist"
    assert extract_job_title("Sales Associate\nRetail store.") == "Sales Associate"
    assert extract_job_title("·  3–5 years of Mern Stack  Developer experience") == "Mern Stack Developer"


def test_jd_headings_and_eg_are_not_requirements():
    labels = {r.label for r in build_profile(HR_JD).requirements}
    assert not {"ideal", "what", "e.g. bamboohr"} & labels, labels
    assert "hris / ats" in labels or "hris" in labels


def test_bracket_of_instances_is_one_requirement():
    labels = [r.label for r in build_profile(HR_JD).requirements]
    assert "ms office" in labels
    assert "excel" not in labels and "powerpoint" not in labels
    assert not any("bamboohr" in lbl for lbl in labels)
    py = [r.label for r in build_profile("Requirements:\nPython frameworks (e.g. Django, Flask)").requirements]
    assert "python" in py and "django / flask" in py


def test_cv_titles_skip_lowercase_prose():
    cv = ("EXPERIENCE\nSenior Technical Recruiter, TechHire\nJan 2021 - Present\n"
          "- Screening calls, interview coordination with hiring managers\n")
    titles = extract_job_titles(cv)
    assert "Senior Technical Recruiter" in titles
    assert not any("coordination" in t for t in titles), titles


def test_business_degrees_and_hr_field():
    text = "EDUCATION\nMBA (Human Resources), IBA Karachi, 2019\nBBA, IBA Karachi, 2017"
    assert [d.display() for d in extract_degrees(text)] == ["BBA (2017)", "MBA Human Resources (2019)"]
    assert {"business administration", "human resources"} <= extract_education(text)


def test_hr_jd_ranks_hr_cv_first():
    res = evaluate_batch(HR_JD, [("dev.txt", DEV_CV.encode()), ("hr.txt", HR_CV.encode())])
    hr = next(c for c in res.candidates if c.name == "hr.txt")
    assert hr.rank == 1 and "Highly" in hr.category, (hr.score, hr.notes)
    assert hr.components["title"].value == 1.0
    assert hr.components["education"].value == 1.0


def test_empty_files_are_not_duplicates_of_each_other():
    res = evaluate_batch(HR_JD, [("a.pdf", b""), ("b.txt", b"")])
    assert [c.status for c in res.candidates] == ["rejected", "rejected"]
    assert all("empty" in c.reject_reason for c in res.candidates)


def test_skill_mentions_scale_linearly():
    text = "React, Node.js, MongoDB, Express, Git, REST APIs, HTML, CSS, JavaScript, MySQL.\n" * 2000
    t0 = time.time()
    hits = skill_mentions(text)
    assert time.time() - t0 < 10
    assert len(hits) >= 2000 * 9


def test_very_long_cv_is_truncated_with_warning():
    res = evaluate_batch(HR_JD, [("long.txt", (HR_CV * 400).encode())])
    c = res.candidates[0]
    assert c.status == "scored"
    assert any("Very long document" in w for w in c.warnings)

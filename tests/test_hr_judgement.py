"""Rankings a recruiter would sign off on: relevant experience, graded gates, JD reading."""
from modules.jd_profile import build_profile, detect_seniority, extract_job_title
from modules.pipeline import below_min_experience_cap, evaluate_batch

MOBILE_JD = """Position: Mobile App Developer
Requirements
Minimum 1 year of professional mobile app development experience.
Flutter and Dart, REST APIs and JSON, state management (Provider / BLoC), Firebase, Git.
Android / iOS development principles.
Basic understanding of AI/ML integration (preferred).
"""

FULLSTACK_ML_CV = """Bilal Khan
EXPERIENCE
ICT Consultant, Acme Software — Jan 2020 – Present
Full-stack developer building Laravel and Node.js web apps with MySQL. Built an OCR system
and machine learning models for document processing. REST APIs, JSON, Git.
EDUCATION
BS Software Engineering, 2019
"""

FLUTTER_CV = """Sana Ali
Flutter Developer
EXPERIENCE
Flutter Developer, AppWorks — Jan 2024 – Present
Built Flutter / Dart apps with Provider state management, Firebase and REST APIs (JSON). Git.
EDUCATION
BS Computer Science, 2023
"""


def test_one_incidental_skill_does_not_make_a_job_relevant():
    res = evaluate_batch(MOBILE_JD, [("fs.txt", FULLSTACK_ML_CV.encode()), ("fl.txt", FLUTTER_CV.encode())])
    fs = next(c for c in res.candidates if c.name == "fs.txt")
    fl = next(c for c in res.candidates if c.name == "fl.txt")
    assert fs.relevant_years == 0.0, [(r.header, r.relevance_reason) for r in fs.roles]
    assert fl.relevant_years >= 1.0
    assert fl.rank == 1 and fs.score < fl.score


def test_experience_gate_is_graded_by_shortfall():
    assert below_min_experience_cap(0, 7) == 25
    assert below_min_experience_cap(1.8, 3) > below_min_experience_cap(0.6, 3)
    assert below_min_experience_cap(0, 0.5) > 40


def test_just_under_minimum_is_accepted_and_flagged():
    jd = "Position: Flutter Developer\nRequirements\n2+ years of experience with Flutter, Dart, Firebase and Git."
    cv = FLUTTER_CV.replace("Jan 2024 – Present", "Jan 2024 – Oct 2025")
    c = evaluate_batch(jd, [("fl.txt", cv.encode())]).candidates[0]
    assert any(f.startswith("⏳ Just under minimum") for f in c.flags)
    assert not any("minimum experience" in n for n in c.notes)


def test_capped_candidates_do_not_tie():
    jd = "Position: Senior Flutter Developer\nRequirements\n5+ years of experience with Flutter, Dart and Firebase."
    strong = FLUTTER_CV
    weak = FLUTTER_CV.replace("Provider state management, Firebase and REST APIs (JSON). Git.", "")
    res = evaluate_batch(jd, [("strong.txt", strong.encode()), ("weak.txt", weak.encode())])
    s, w = (next(c for c in res.candidates if c.name == n) for n in ("strong.txt", "weak.txt"))
    assert s.score > w.score


def test_degree_subjects_are_not_skill_requirements():
    p = build_profile("Account Manager\nRequirements:\n- 3+ years of experience in account management, sales or "
                      "customer success.\n- Proven track record of meeting sales targets and upselling.\n"
                      "- Bachelor's degree in Business, Marketing or a related field.")
    labels = {r.label for r in p.requirements}
    assert "marketing" not in labels and "sales" in labels and "upselling" in labels, labels
    assert {"business administration", "marketing"} <= p.education_fields
    assert p.seniority == "mid"


def test_jd_concepts_written_in_words():
    p = build_profile("Backend Developer\nTechnical Skills:\nUnderstanding of Java, Python, or Node.js\n"
                      "Solid understanding of Backend Development Concepts\nKnowledge of Databases & Server-Side Logic")
    labels = {r.label for r in p.requirements}
    assert {"backend development", "database"} <= labels, labels
    p = build_profile("AI ENGINEER\nMust have hands on experience with RAG, Vector Database and open source LLMs.\n"
                      "Experience of backend API development in Python frameworks such as Fast and/or Flask is a must.")
    labels = {r.label for r in p.requirements}
    assert "fastapi / flask" in labels, labels
    assert not {"database", "open source", "backend development"} & labels, labels
    p = build_profile("JAVA ENGINEER\n- Strong expertise in distributed systems design.\n"
                      "- Core contributions to open-source projects.")
    assert {"distributed systems", "open source"} <= {r.label for r in p.requirements}


def test_job_title_inside_a_sentence_and_all_caps():
    assert extract_job_title("SENIOR SOFTWARE ARCHITECTURE\n\nWe're seeking a Senior Software Architect to "
                             "design large-scale backend systems.") == "Senior Software Architect"
    assert extract_job_title("SENIOR DATA SCIENTIST\nAbout the role") == "Senior Data Scientist"
    assert extract_job_title("MERN STACK DEVELOPER\n") == "MERN Stack Developer"
    assert detect_seniority("", "Senior Software Architect") == "senior"


def test_familiarity_line_is_not_a_must_have():
    p = build_profile("Position: Backend Engineer\nRequirements:\nPython and Django.\n"
                      "Familiarity with AI-assisted development tools (e.g., Copilot, ChatGPT).")
    imp = {r.label: r.importance for r in p.requirements}
    assert imp.get("python") == "must" and imp.get("openai") == "normal", imp


def test_undated_relevant_job_is_flagged_not_counted():
    cv = ("Omar Shah\nEXPERIENCE\nWeb Developer\nAcme | Jan 2022 – Dec 2023\nBuilt PHP and MySQL web apps.\n"
          "Flutter Developer Intern\nBuilt Flutter / Dart apps with Provider and Firebase.\n"
          "EDUCATION\nBS Computer Science, 2021\n")
    c = evaluate_batch(MOBILE_JD, [("omar.txt", cv.encode())]).candidates[0]
    assert c.relevant_years == 0.0 and c.total_years == 2.0
    assert [u.header for u in c.undated_roles] == ["Flutter Developer Intern"] and c.undated_roles[0].relevant
    assert any("without dates" in f for f in c.flags)
    assert any("no dates given" in n and "Flutter Developer Intern" in n for n in c.notes)


def test_placeholder_dates_add_no_experience():
    cv = "Ravi Kumar\nEXPERIENCE\n0, 0 Jan 2026 - Jan 2026\nSeeking junior roles.\nEDUCATION\nBS CS, 2025\n"
    c = evaluate_batch(MOBILE_JD, [("ravi.txt", cv.encode())]).candidates[0]
    assert c.total_years == 0 and not c.roles
    assert any("placeholder" in n for n in c.notes)


def test_inferred_knowledge_counts_less_than_hands_on_work_for_relevant_years():
    jd = ("Position: LLM Engineer\nRequirements\n2+ years of experience.\nLLMs and Generative AI.\n"
          "Prompt engineering.\nPython.\n")
    rag = ("Ali\nEXPERIENCE\nAI Engineer, Acme — Jan 2022 – Dec 2023\nBuilt a RAG assistant in Python.\n"
           "EDUCATION\nBS CS, 2021\n")
    hf = ("Zara\nEXPERIENCE\nData Analyst, Acme — Jan 2022 – Dec 2023\nUsed HuggingFace datasets for reporting.\n"
          "EDUCATION\nBS CS, 2021\n")
    res = evaluate_batch(jd, [("rag.txt", rag.encode()), ("hf.txt", hf.encode())])
    r, h = (next(c for c in res.candidates if c.name == n) for n in ("rag.txt", "hf.txt"))
    assert r.relevant_years == 2.0, [(x.header, x.relevance_reason) for x in r.roles]
    assert h.relevant_years == 0.0
    assert r.rank == 1


def test_more_relevant_experience_ranks_higher_with_equal_skills():
    junior = FLUTTER_CV
    senior = FLUTTER_CV.replace("Jan 2024 – Present", "Jan 2020 – Present")
    for jd in ("Position: Flutter Developer\nRequirements\n1+ years of experience with Flutter, Dart, Firebase and Git.",
               "Position: Flutter Developer\nRequirements\nFlutter, Dart, Firebase and Git."):
        res = evaluate_batch(jd, [("junior.txt", junior.encode()), ("senior.txt", senior.encode())])
        j, s = (next(c for c in res.candidates if c.name == n) for n in ("junior.txt", "senior.txt"))
        assert s.relevant_years > j.relevant_years and s.rank == 1 and s.score > j.score, jd


def test_fresher_role_does_not_reward_extra_years():
    jd = "Position: Flutter Developer (Trainee)\nRequirements\nFlutter, Dart, Firebase and Git."
    senior = FLUTTER_CV.replace("Jan 2024 – Present", "Jan 2020 – Present")
    res = evaluate_batch(jd, [("junior.txt", FLUTTER_CV.encode()), ("senior.txt", senior.encode())])
    j, s = (next(c for c in res.candidates if c.name == n) for n in ("junior.txt", "senior.txt"))
    assert s.score == j.score

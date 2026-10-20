from datetime import date

import pytest
from conftest import read, sample_cvs

from modules.education import extract_degrees, has_bachelors
from modules.experience import analyse_experience
from modules.extractor import extract_text


def cv_text(name_part):
    found=[p for p in sample_cvs() if name_part in p]
    if not found:
        pytest.skip("Private original resume fixture excluded")
    return extract_text(read(found[0]), "pdf")


@pytest.mark.parametrize("text,years", [
    ("WORK EXPERIENCE\nDeveloper Jan 2020 to Dec 2021\n", 2.0),
    ("WORK EXPERIENCE\nDeveloper Sept 2019 – Aug 2020\n", 1.0),
    ("WORK EXPERIENCE\nAnalyst 1995 - 1999\n", 4.0),
    ("WORK EXPERIENCE\nDev 06/2021 - 05/2022\n", 1.0),
    ("WORK EXPERIENCE\nDev [ 01/02/2019 – 31/01/2020 ]\n", 1.0),
    ("WORK EXPERIENCE\nDev Jan 2020 - Dec 2020\nLead Jun 2020 - Dec 2021\n", 2.0),
    ("WORK EXPERIENCE\nDev 2025 (Jan – Jun)\n", 0.5),
    ("WORK EXPERIENCE\nDev July(2023) – Dec(2023)\n", 0.5),
])
def test_date_formats(text, years):
    assert analyse_experience(text).total_years == years


def test_present_counts_to_today():
    y = date.today().year - 2
    r = analyse_experience(f"WORK EXPERIENCE\nEngineer Jan {y} – Present\n")
    assert 1.7 <= r.total_years <= 2.9


def test_future_and_absurd_ranges_ignored():
    y = date.today().year + 2
    assert analyse_experience(f"WORK EXPERIENCE\nDev Jan {y} - Dec {y + 1}\n").total_years == 0


def test_academic_projects_not_counted():
    r = analyse_experience(cv_text("Rasool Bux"))
    assert r.total_years == 0.0
    assert any("academic" in why for _, why in r.excluded)


def test_preamble_jobs_not_lost_when_heading_undetected():
    r = analyse_experience(cv_text("Account-Manager"))
    assert r.total_years >= 9.0 and len(r.roles) == 3


@pytest.mark.parametrize("line", [
    "0, 0 Jan 2026 - Jan 2026",
    "MM/YYYY Jan 2022 – Mar 2023",
    "20XX: Jan 2019 - Jan 2020",
])
def test_template_placeholder_dates_are_not_jobs(line):
    r = analyse_experience(f"WORK EXPERIENCE\nSoftware Engineer\n{line}\nBuilt things.\n")
    assert r.total_years == 0 and not r.roles
    assert any("placeholder" in why for _, why in r.excluded)


def test_mm_alam_road_is_not_a_placeholder():
    r = analyse_experience("WORK EXPERIENCE\nDeveloper, Acme, MM Alam Road, Lahore | Jan 2020 - Dec 2021\n")
    assert r.total_years == 2.0


def test_placeholder_date_cv_has_no_experience():
    r = analyse_experience(cv_text("Ravindar"))
    assert r.total_years == 0 and not r.roles


UNDATED = ("WORK EXPERIENCE\n"
           "Web Developer\nAcme | Jan 2020 – Dec 2020\n"
           "Built PHP and MySQL web apps.\n"
           "Flutter Developer Intern\n"
           "Built two Flutter apps with Firebase.\n"
           "Java Developer\nGlobex | Jan 2018 – Jun 2018\n"
           "Built Java desktop tools.\n")


def test_undated_job_is_split_from_the_dated_job_above():
    r = analyse_experience(UNDATED)
    assert [ro.months for ro in r.roles] == [12, 6]
    assert "Flutter" not in r.roles[0].text
    assert [u.header for u in r.undated] == ["Flutter Developer Intern"]
    assert "Firebase" in r.undated[0].text
    assert r.total_years == 1.5


def test_description_lines_are_not_mistaken_for_undated_jobs():
    text = ("WORK EXPERIENCE\nSenior Developer\nAcme | Jan 2020 – Dec 2021\n"
            "Led a team of 5 developers.\n• Mentored junior engineers\nWorked as Scrum Master for the team\n"
            "Responsible for Developer onboarding\n")
    r = analyse_experience(text)
    assert not r.undated and "onboarding" in r.roles[0].text


def test_daniyal_undated_internship_and_company_lines():
    r = analyse_experience(cv_text("Daniyal"))
    assert len(r.roles) == 3
    zeeshsoft = r.roles[1]
    assert "Zeeshsoft" in zeeshsoft.header and "Flutter" not in zeeshsoft.text
    assert "Gexton" not in zeeshsoft.text and "Gexton" in r.roles[2].header
    assert any("Flutter developer" in u.header for u in r.undated)
    assert any("Teacher" in u.header for u in r.undated)


def test_two_column_heading_before_its_date_is_not_undated():
    r = analyse_experience(cv_text("Copenhagen"))
    assert len(r.roles) == 2 and not r.undated
    assert "Gladstone" in r.roles[1].header and "Gladstone" not in r.roles[0].text


def test_education_dates_not_counted_as_jobs():
    assert analyse_experience(cv_text("Muhammad Imran")).total_years == pytest.approx(2.2, abs=0.2)
    moscow = analyse_experience(cv_text("Moscow"))
    assert len(moscow.roles) == 2


def test_three_degrees_each_with_year():
    degs = extract_degrees(cv_text("Qasid"))
    assert [(d.label, d.year) for d in degs] == [("BSc", 2018), ("MSc", 2020), ("MPhil", 2023)]
    assert "Space Science" in degs[0].field and "Computational Astrophysics" in degs[2].field


def test_in_progress_masters_and_completed_bachelors():
    degs = extract_degrees(cv_text("Daniyal"))
    assert [(d.level, d.year, d.in_progress) for d in degs][0] == ("bachelors", 2021, False)
    assert degs[1].level == "masters" and degs[1].in_progress


@pytest.mark.parametrize("name,year", [("Account-Manager", 2017), ("Ravindar", 2025),
                                        ("Muhammad Ramzan", 2024), ("Copenhagen", 2021)])
def test_degree_year_not_borrowed_from_neighbouring_entry(name, year):
    assert extract_degrees(cv_text(name))[0].year == year


@pytest.mark.parametrize("word,ok", [("backlogs", False), ("backers", False), ("Bachir", False),
                                     ("Bechelor", True), ("Batchelors", True), ("Bachelor", True)])
def test_bachelor_typo_matcher(word, ok):
    assert has_bachelors(f"Managed product {word} in Computer Science") == ok


def test_no_degree_cvs():
    for name in ("Dublin", "AbdulRehman"):
        assert extract_degrees(cv_text(name)) == []

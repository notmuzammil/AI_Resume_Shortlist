"""
Parses a Job Description into an explicit requirement profile — once.

Key ideas:
  * A JD is a list of REQUIREMENTS, not a bag of skills. Alternatives
    written as "Git or SVN", "Django/Flask", "TensorFlow, HuggingFace,
    or equivalent", "such as Hadoop, Spark, or SQL" form ONE requirement
    that any member satisfies. Previously each alternative was a
    separate mandatory skill, so a candidate with Git but not SVN lost
    points (and SVN could even be a must-have).
  * Each requirement has an importance: must / normal / nice, from the
    section it sits in ("Requirements" vs "Preferred") or inline markers
    ("is a must", "(preferred)", "is a plus").
  * Experience is a MINIMUM: "6 months" → 0.5 years, "3+ years" → 3.
  * Seniority comes from the TITLE. Body words like "lead the
    development" or "report to the manager" no longer turn a JD into a
    lead-level (8-year) role.
"""
import re
from dataclasses import dataclass, field

from config import MUST_HAVE_MAX_SHARE, SENIORITY_YEARS
from modules.education import extract_education
from modules.skill_extractor import SOFT_SKILLS, canonicalize_skill, extract_skills, skill_mentions

ROLE_NOUNS = (
    r'engineer|developer|programmer|scientist|analyst|designer|'
    r'manager|architect|administrator|consultant|specialist|'
    r'intern|trainee|officer|executive|lead|researcher|tester|'
    r'generalist|recruiter|coordinator|accountant|representative|supervisor|director|advisor|'
    r'assistant|associate(?!\s+(?:degree|of\b|in\b))|business\s+partner'
)

ROLE_FAMILIES = {
    "engineer": "build", "developer": "build", "programmer": "build",
    "architect": "build", "tester": "build",
    "scientist": "analyse", "analyst": "analyse", "researcher": "analyse",
    "manager": "manage", "lead": "manage", "executive": "manage",
    "officer": "manage", "supervisor": "manage", "director": "manage", "partner": "manage",
    "specialist": "operate",
    "designer": "design",
    "intern": "junior", "trainee": "junior",
    "generalist": "operate", "recruiter": "operate", "coordinator": "operate", "accountant": "operate",
    "representative": "operate", "advisor": "operate", "assistant": "operate", "associate": "operate",
}

_TITLE_LABEL_RE = re.compile(
    r'(?:^|\n)[^\S\n]*(?:position|job\s*title|role|designation|vacancy|job)[^\S\n]*'
    r'[:\-][^\S\n]*(.+)', re.IGNORECASE
)
_TITLE_INLINE_RE = re.compile(
    r'\b([A-Za-z][A-Za-z+/#.\- ]{1,40}?\b(?:' + ROLE_NOUNS + r')s?)\b',
    re.IGNORECASE
)

_TITLE_NOISE = {
    "senior", "sr", "junior", "jr", "lead", "principal", "staff",
    "trainee", "intern", "internship", "entry", "level", "mid",
    "associate", "assistant", "head", "chief", "i", "ii", "iii",
    "full", "time", "part", "remote", "onsite", "hybrid", "contract",
    "position", "role", "job", "vacancy", "opening", "the", "a", "an",
    "of", "for", "and", "in", "at", "looking", "hiring", "seeking", "we",
    "are", "freelance", "passionate", "experienced", "is", "to", "join",
    "years", "year", "yrs", "experience", "exp", "as", "with", "urgent", "urgently", "wanted", "needed",
}

_TITLE_SENIORITY = [
    ("entry", r'\b(intern(?:ship)?|trainee|graduate\s+program|fresher|entry[\s-]level|apprentice)\b'),
    ("lead", r'\b(lead|principal|staff|head\s+of|director|chief)\b'),
    ("senior", r'\b(senior|sr\.?|architect)\b'),
    ("junior", r'\b(junior|jr\.?|associate)\b'),
]
_BODY_ENTRY_RE = re.compile(
    r'\b(?:entry[\s-]level\s+(?:role|position|job)|this\s+is\s+an?\s+(?:internship|trainee)|'
    r'no\s+(?:prior\s+)?experience\s+(?:is\s+)?required|internship\s+program|trainee\s+program)\b',
    re.IGNORECASE)
_BODY_SENIOR_RE = re.compile(r'\b(?:senior[\s-]level|senior\s+(?:role|position))\b', re.IGNORECASE)

_FRESHER_PREFERRED_RE = re.compile(
    r'\bfresh\s+graduates?\b|\bfreshers?\b|\brecent\s+graduates?\b|'
    r'\bno\s+(?:prior\s+)?experience\s+(?:is\s+)?required\b|\bentry[\s-]level\b',
    re.IGNORECASE
)

_FRESHER_ELIGIBLE_ONLY_RE = re.compile(
    r'\b(?:eligible|welcome[ds]?|can\s+(?:also\s+)?apply|may\s+(?:also\s+)?apply|encouraged\s+to\s+apply|'
    r'also\s+(?:be\s+)?considered)\b', re.IGNORECASE)


def is_fresher_role(jd_text: str, seniority: str) -> bool:
    """An entry-level role, or one that PREFERS freshers — not one that merely accepts them."""
    if seniority == "entry":
        return True
    for m in _FRESHER_PREFERRED_RE.finditer(jd_text):
        ls = jd_text.rfind("\n", 0, m.start()) + 1
        le = jd_text.find("\n", m.end())
        if not _FRESHER_ELIGIBLE_ONLY_RE.search(jd_text[ls:le if le != -1 else len(jd_text)]):
            return True
    return False


_NUM = r'(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve)'
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
_EXP_WORDS = r'(?:relevant\s+|professional\s+|industry\s+|hands[\s-]on\s+|work(?:ing)?\s+|commercial\s+|practical\s+)*'
_REQUIRED_YEARS_PATTERNS = [
    (r'(?:minimum\s+(?:of\s+)?|at\s+least\s+|min\.?\s+)?' + _NUM +
     r'\s*\+?\s*(?:(?:-|–|to)\s*\d+(?:\.\d+)?\s*)?(years?|yrs?|months?)\b'
     r'(?=[^.\n]{0,60}?\bexperience\b|\s*(?:of\s+)?' + _EXP_WORDS + r'(?:experience|exp)\b)', 1.0),
    (r'experience\s*(?:of\s*)?(?::|-)?\s*(?:minimum\s+|at\s+least\s+)?' + _NUM +
     r'\s*\+?\s*(years?|yrs?|months?)\b', 1.0),
    (r'(?:minimum|at\s+least)\s+(?:of\s+)?' + _NUM + r'\s*\+?\s*(years?|yrs?|months?)\b', 1.0),
    (r'\(\s*' + _NUM + r'\s*\+\s*(years?|yrs?)\s*\)', 1.0),
    (r'(?<![\d.])' + _NUM + r'\s*\+\s*(years?|yrs?)\b', 1.0),
]

_DEGREE_YEARS_RE = re.compile(
    r'(?:bachelor|master|degree|bs|ms|bsc|msc|be|phd|education|'
    r'qualification|schooling)[^.\n]{0,40}?\(?\s*\d+\s*years?(?:\s+of\s+education)?\s*\)?',
    re.IGNORECASE
)


def _to_num(tok: str) -> float:
    tok = tok.lower()
    return float(_WORD_NUM[tok]) if tok in _WORD_NUM else float(tok)


_COMPANY_RE = re.compile(r'\b(?:we|we\'ve|we\'re|our|us|company|firm|agency|founded|established|serving|'
                         r'clients|customers|in\s+business|track\s+record|organization|organisation)\b')
_CANDIDATE_RE = re.compile(r'\b(?:you|your|candidate|applicant|must|required?|requirements?|minimum|'
                           r'at\s+least|should|need|looking\s+for|seeking|ideal)\b')


def _is_company_boast(text: str, pos: int) -> bool:
    """"With 20+ years of experience, our company ..." describes the employer."""
    start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos)) + 1
    ends = [e for e in (text.find(".", pos), text.find("\n", pos)) if e != -1]
    sentence = text[start:min(ends) if ends else len(text)]
    return bool(_COMPANY_RE.search(sentence)) and not _CANDIDATE_RE.search(sentence)


def extract_required_years(jd_text: str) -> float:
    """
    Minimum years of experience the JD asks for (0.0 if none). Months
    are converted ("6 months" → 0.5). When several figures appear the
    LARGEST stated minimum wins (e.g. "5+ years overall, 2+ in Go").
    """
    cleaned = _DEGREE_YEARS_RE.sub(' ', jd_text.lower())
    found = []
    for pat, _ in _REQUIRED_YEARS_PATTERNS:
        for m in re.finditer(pat, cleaned):
            if _is_company_boast(cleaned, m.start()):
                continue
            try:
                value = _to_num(m.group(1))
            except (ValueError, KeyError):
                continue
            unit = m.group(2)
            if unit.startswith("month"):
                value = value / 12.0
            if 0 < value <= 30:
                found.append(value)
    return round(max(found), 2) if found else 0.0


_TITLE_SENTENCE_RE = re.compile(
    r"(?i:\b(?:seeking|looking\s+for|hiring|recruiting)\s+(?:an?\s+|our\s+next\s+)?"
    r"(?:talented\s+|experienced\s+|passionate\s+|skilled\s+)?)"
    r"((?:[A-Z][\w+#./-]*\s+){0,4}?(?i:" + ROLE_NOUNS + r"))s?\b")
_KEEP_UPPER = {"AI", "ML", "HR", "BI", "IT", "QA", "SQA", "UI", "UX", "UI/UX", "MERN", "MEAN", "LAMP", "SEO",
               "SAP", "ERP", "CRM", "PHP", "SQL", "AWS", "GCP", "NLP", "LLM", "MLOPS", "DEVOPS", "IOS", ".NET",
               "C#", "C++", "R&D", "SRE", "DBA"}
_CASED = {"MLOPS": "MLOps", "DEVOPS": "DevOps", "IOS": "iOS"}


def _display_title(title: str) -> str:
    """'SENIOR DATA SCIENTIST' → 'Senior Data Scientist' (acronyms kept)."""
    if not title or any(ch.islower() for ch in title):
        return title
    out = []
    for w in title.split():
        up = w.upper()
        out.append(_CASED.get(up, up) if up in _KEEP_UPPER else w.capitalize())
    return " ".join(out)


def _clean_title(title: str) -> str:
    """"Looking for passionate Java Developers" → "Java Developers"."""
    parts = title.split()
    words = [w for i, w in enumerate(parts) if w.lower() not in _TITLE_NOISE or
             w.lower() in ("senior", "junior", "lead", "trainee", "intern", "principal", "staff") or
             (i == len(parts) - 1 and w.lower() in ("associate", "assistant"))]
    return _display_title(" ".join(words)[:80] or title[:80])


def extract_job_title(jd_text: str) -> str:
    m = _TITLE_LABEL_RE.search(jd_text)
    if m:
        line = re.split(r'[|\n]|\s{3,}', m.group(1).strip())[0].strip()
        if line:
            return _display_title(line[:80])
    head_lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()][:6]
    for ln in head_lines:
        mm = _TITLE_SENTENCE_RE.search(ln)
        if mm:
            return _clean_title(mm.group(1).strip())
        mm = _TITLE_INLINE_RE.search(ln)
        if mm and len(ln) <= 120:
            return _clean_title(mm.group(1).strip())
    return ""


def detect_seniority(jd_text: str, title: str = "") -> str:
    """entry / junior / mid / senior / lead — from the title first."""
    for level, pattern in _TITLE_SENIORITY:
        if title and re.search(pattern, title, re.IGNORECASE):
            return level
    if _BODY_ENTRY_RE.search(jd_text):
        return "entry"
    if _BODY_SENIOR_RE.search(jd_text):
        return "senior"
    return "mid"


def title_tokens(title: str) -> list:
    if not title:
        return []
    title = re.sub(r'\([^)]*\)', ' ', title)
    words = re.findall(r"[A-Za-z][A-Za-z+#.]*", title.lower())
    out = []
    for i, w in enumerate(words):
        w = w.rstrip(".")
        if w.endswith("s") and w[:-1] in ROLE_FAMILIES:
            w = w[:-1]
        if (w not in _TITLE_NOISE or (i == len(words) - 1 and w in ("associate", "assistant"))) and len(w) > 1:
            out.append(w)
    return out


def requires_graduates_only(jd_text: str) -> bool:
    return bool(re.search(
        r'graduat(?:ed|es)\s+(?:students?\s+)?only'
        r'|must\s+(?:have\s+)?(?:already\s+)?graduated'
        r'|already\s+graduated'
        r'|completed\s+(?:their\s+)?degree\s+(?:is\s+)?required'
        r'|no\s+longer\s+enrolled',
        jd_text, re.IGNORECASE
    ))


_BACHELOR_REQ_RE = re.compile(
    r"bachelor'?s?\s+(?:or\s+master'?s?\s+)?degree"
    r"|bachelor'?s?\s+(?:in|of)\b"
    r"|b\.?s\.?c?\.?\s+(?:degree|required)"
    r"|undergraduate\s+degree\s+required"
    r"|4[- ]year\s+degree"
    r"|degree\s+(?:is\s+)?required"
    r"|minimum\s+(?:qualification|education)\s*:?\s*bachelor",
    re.IGNORECASE)
_SOFT_LINE_RE = re.compile(r'\b(?:preferred|is\s+a\s+plus|a\s+plus|advantage(?:ous)?|desirable|'
                           r'nice\s+to\s+have|good\s+to\s+have|bonus|optional|or\s+equivalent\s+experience)\b',
                           re.IGNORECASE)


def requires_bachelors(jd_text: str) -> bool:
    """
    True only when a bachelor's degree is stated as a requirement. A
    line like "Bachelor's or Master's degree … preferred" is a
    preference and must not hard-reject candidates.
    """
    for line in jd_text.splitlines():
        if _BACHELOR_REQ_RE.search(line) and not _SOFT_LINE_RE.search(line):
            return True
    return False


@dataclass
class Requirement:
    """
    One JD requirement.

    items      — tuple of alternative-sets, e.g. (("git", "svn"),) for
                 "Git or SVN", or (("mongodb",), ("mysql",), ("sql server",))
                 for an EXAMPLES list "databases (MongoDB, MySQL, SQL Server)".
    min_hits   — how many items must be shown for full credit (1 for a
                 single skill / alternatives; up to 2 for an examples list).
    """
    skills: tuple
    importance: str = "normal"
    source: str = ""
    items: tuple = ()
    min_hits: int = 1

    def __post_init__(self):
        if not self.items:
            self.items = (tuple(self.skills),)

    @property
    def label(self) -> str:
        if len(self.items) == 1:
            return " / ".join(self.items[0])
        names = ", ".join(" / ".join(i) for i in self.items)
        return f"{names} (any {self.min_hits})" if self.min_hits < len(self.items) else names


_REQUIRED_HEADING_RE = re.compile(
    r'^[^\w\n]*(?:required|requirements?|must[\s-]?haves?|mandatory|essential|'
    r'(?:basic\s+|minimum\s+)?qualifications?|skills?\s+required|required\s+skills?|'
    r'what\s+you\s+(?:need|bring|will\s+need)|who\s+you\s+are)\b[^\n]{0,40}$',
    re.IGNORECASE | re.MULTILINE)
_PREFERRED_HEADING_RE = re.compile(
    r'^[^\w\n]*(?:preferred|nice[\s-]?to[\s-]?have|bonus|good\s+to\s+have|'
    r'desirable|advantages?|plus(?:es)?|preferred\s+(?:skills?|qualifications?))\b[^\n]{0,40}$',
    re.IGNORECASE | re.MULTILINE)
_OTHER_HEADING_RE = re.compile(
    r'^[^\w\n]*(?:responsibilities|what\s+you\'?ll\s+do|duties|about\s+(?:us|the\s+role)|benefits|'
    r'key\s+responsibilities|role\s+overview|job\s+description|perks)\b[^\n]{0,40}$',
    re.IGNORECASE | re.MULTILINE)

_INLINE_MUST_RE = re.compile(r'\b(?:must|mandatory|required|essential|is\s+a\s+must)\b', re.IGNORECASE)
_INLINE_NICE_RE = re.compile(r'\b(?:preferred|is\s+a\s+plus|a\s+plus|nice\s+to\s+have|good\s+to\s+have|'
                             r'bonus|desirable|advantage(?:ous)?|optional)\b|\(\s*plus\s*\)', re.IGNORECASE)

_LIST_GAP_RE = re.compile(r'^[\s,;/&]*(?:\b(?:and/or|or|and)\b)?[\s,;/&]*$', re.IGNORECASE)
_OR_GAP_RE = re.compile(r'/|\bor\b', re.IGNORECASE)
_OR_TAIL_RE = re.compile(r'^\s*[,)]?\s*(?:(?:and/)?or\s+(?:similar|equivalent|other|related|comparable|any)|'
                         r'etc\.?|or\s+equivalent|\.\.\.)', re.IGNORECASE)
_OR_HEAD_RE = re.compile(r'(?:such\s+as|e\.\s?g\.?|for\s+example|like|i\.e\.|either|any\s+of|one\s+of)\s*[:(]?\s*$',
                         re.IGNORECASE)


_BRACKET_AFTER_SKILL_RE = re.compile(r'[^\S\n]*(?:[A-Za-z]+[^\S\n]+){0,2}\(')


def _strip_parens(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'\([^()]*\)', '', s)
    return s


def _line_requirements(line: str) -> list:
    """
    Group the skill mentions of one JD line into requirements.
    Returns [(tuple_of_skills, [(start,end) of mentions])].
    """
    mentions = skill_mentions(line)
    if not mentions:
        return []

    absorbed = set()
    for canon, _s, e in mentions:
        if canon in ("mern", "mean", "lamp", "full stack") and line[e:e + 3].lstrip().startswith("("):
            close = line.find(")", e)
            if close != -1:
                for j, (_, s2, e2) in enumerate(mentions):
                    if s2 > e and e2 <= close:
                        absorbed.add(j)

    from modules.matcher import isa_instances
    for idx, (canon, _s, e) in enumerate(mentions):
        if idx in absorbed or not _BRACKET_AFTER_SKILL_RE.match(line, e):
            continue
        open_ = line.find("(", e)
        close = line.find(")", open_)
        inside = [j for j, (_, s2, e2) in enumerate(mentions) if open_ < s2 and (close == -1 or e2 <= close)]
        cats, k = {canon}, idx
        while k > 0 and _LIST_GAP_RE.match(_strip_parens(line[mentions[k - 1][2]:mentions[k][1]])):
            k -= 1
            cats.add(mentions[k][0])
        instances = set().union(*(isa_instances(c) for c in cats))
        if inside and all(mentions[j][0] in instances for j in inside):
            absorbed.update(inside)
    mentions = [m for j, m in enumerate(mentions) if j not in absorbed]

    runs = [[mentions[0]]]
    gaps = [[]]
    for prev, cur in zip(mentions, mentions[1:]):
        gap = line[prev[2]:cur[1]]
        gap_np = _strip_parens(gap)
        if gap.count("(") == gap.count(")") and _LIST_GAP_RE.match(gap_np):
            runs[-1].append(cur)
            if re.search(r'\bor\b', gap_np, re.IGNORECASE):
                gaps[-1].append("or")
            elif "/" in gap_np:
                gaps[-1].append("/")
            else:
                gaps[-1].append("and")
        else:
            runs.append([cur])
            gaps.append([])

    out = []
    for run, kinds in zip(runs, gaps):
        start, end = run[0][1], run[-1][2]
        whole_or = "or" in kinds or (len(run) > 1 and bool(
            _OR_TAIL_RE.match(line[end:end + 30]) or _OR_HEAD_RE.search(line[max(0, start - 25):start])))
        if whole_or:
            groups = [run]
        else:
            groups = [[run[0]]]
            for kind, m in zip(kinds, run[1:]):
                if kind == "/":
                    groups[-1].append(m)
                else:
                    groups.append([m])
        before = line[:start].rstrip()
        after = line[end:].lstrip()
        prev_mention_end = max((e for _, _, e in mentions if e <= start), default=0)
        is_examples = (len(groups) >= 2 and before.endswith("(") and after.startswith(")")
                       and prev_mention_end < len(before) - 1
                       and line[prev_mention_end:len(before) - 1].strip() != "")
        if is_examples:
            items = tuple(tuple(dict.fromkeys(c for c, _, _ in g)) for g in groups)
            flat = tuple(dict.fromkeys(x for it in items for x in it))
            out.append((flat, [(s, e) for _, s, e in run], items, min(2, len(items))))
            continue
        for g in groups:
            skills = tuple(dict.fromkeys(c for c, _, _ in g))
            out.append((skills, [(s, e) for _, s, e in g], (skills,), 1))
    return out


def _inline_importance(line: str, spans, line_spans) -> str:
    """
    Importance markers written on the line itself.
      "3+ years of Python (Django/Flask preferred)" → only Django/Flask are nice
      "AWS/OpenStack (Plus)"                         → whole line is nice
      "... stable diffusion is a plus."              → whole line is nice
      "Fast and/or Flask is a must."                 → must
    """
    for m in _INLINE_NICE_RE.finditer(line):
        open_ = line.rfind("(", 0, m.start())
        in_bracket = open_ != -1 and line.rfind(")", 0, m.start()) < open_
        if in_bracket:
            close_ = line.find(")", m.end())
            close_ = len(line) if close_ == -1 else close_
            bracket_skills = [sp for sp in line_spans if open_ < sp[0] and sp[1] <= close_]
            if bracket_skills:
                if any(sp in bracket_skills for sp in spans):
                    return "nice"
                continue
        return "nice"
    if _INLINE_MUST_RE.search(line):
        return "must"
    return ""


def _section_map(jd_text: str) -> list:
    """[(start_offset, kind)] with kind in required/preferred/other."""
    marks = []
    for rx, kind in ((_REQUIRED_HEADING_RE, "required"), (_PREFERRED_HEADING_RE, "preferred"),
                     (_OTHER_HEADING_RE, "other")):
        for m in rx.finditer(jd_text):
            marks.append((m.start(), kind))
    marks.sort()
    return marks


def _section_at(marks, pos: int) -> str:
    kind = ""
    for start, k in marks:
        if start <= pos:
            kind = k
        else:
            break
    return kind


_TEAM_LINE_RE = re.compile(r'\b(?:work|collaborat\w*|coordinat\w*|liaise|partner)\s+(?:closely\s+)?with\b'
                           r'[^.\n]{0,80}\b(?:teams?|designers?|stakeholders?|engineers?|developers?)\b',
                           re.IGNORECASE)


_HEADING_LINE_RE = re.compile(r"^[^a-z,/|•]*[A-Z]{2}[^a-z,/|•]*$")


def _is_heading_line(line: str) -> bool:
    """Short ALL-CAPS line without list punctuation: "IDEAL CANDIDATE", "WHAT WE OFFER"."""
    return bool(_HEADING_LINE_RE.match(line)) and len(line.split()) <= 6


def _drop_team_clause(line: str) -> str:
    m = _TEAM_LINE_RE.search(line)
    return line[:m.start()] if m else line


_DEGREE_SUBJECT_RE = re.compile(
    r"\b(?:bachelor'?s?|master'?s?|degree|diploma|b\.?sc?|m\.?sc?|mba|bba|ph\.?d|graduate)\b"
    r"(?:\s+degree)?\s+(?:in|of)\b[^.;\n]*", re.IGNORECASE)
_DATABASES_RE = re.compile(r"\bdatabases?\b(?!\s+(?:design|administration|tables?|records?|rows?|entries))", re.IGNORECASE)
_OPEN_SOURCE_RE = re.compile(
    r"\bopen[\s-]source\s+(?:contributions?|projects?|development|community|communities|software\s+development)\b"
    r"|\bcontribut\w*\s+to\s+(?:\w+\s+){0,2}open[\s-]source\b", re.IGNORECASE)
_LOW_BAR_RE = re.compile(r"^[\W\d]*(?:familiarity|familiar|exposure|awareness|basic\s+(?:understanding|knowledge)"
                         r"|some\s+(?:experience|knowledge))\b", re.IGNORECASE)
_ADJECTIVE_END_RE = re.compile(r"\s*(?:services?|developers?|engineers?|teams?|apis?|integration|pages?|screens?|tables?|code|repos?(?:itor(?:y|ies))?)\b", re.IGNORECASE)


def _line_mentions_outside(line: str, spans) -> str:
    """The line with the given character spans blanked out."""
    chars = list(line)
    for a, b in spans:
        for i in range(a, min(b, len(chars))):
            chars[i] = " "
    return "".join(chars)


def extract_requirements(jd_text: str, title: str = "") -> list:
    marks = _section_map(jd_text)
    title_line_done = False
    mentioned = set()
    reqs = {}
    order = []
    offset = 0
    for line in jd_text.split("\n"):
        line_start = offset
        offset += len(line) + 1
        if not line.strip():
            continue
        section = _section_at(marks, line_start)
        if title and not title_line_done and title.lower()[:30] in line.lower():
            title_line_done = True
            continue
        mentioned.update(c for c, _, _ in skill_mentions(line))
        line = _drop_team_clause(line)
        line = _line_mentions_outside(line, [m.span() for m in _DEGREE_SUBJECT_RE.finditer(line)])
        if not line.strip():
            continue
        line_reqs = _line_requirements(line)
        spans_on_line = [sp for _, sps, _, _ in line_reqs for sp in sps]
        line_reqs = [r for r in line_reqs if not (
            r[0] in (("backend development",), ("frontend development",))
            and len(line[r[1][0][0]:r[1][0][1]].split()) == 1
            and (any(0 <= s - r[1][0][1] <= 1 for s, _ in spans_on_line)
                 or _ADJECTIVE_END_RE.match(line, r[1][0][1])))]
        db = _DATABASES_RE.search(line)
        if db and not any(a < db.end() and db.start() < b for a, b in spans_on_line):
            from modules.matcher import isa_instances
            named = isa_instances("database")
            if not any(s in named for sk, *_ in line_reqs for s in sk):
                line_reqs.append((("database",), [db.span()], (("database",),), 1))
        oss = _OPEN_SOURCE_RE.search(line)
        if oss:
            line_reqs.append((("open source",), [oss.span()], (("open source",),), 1))
        for sk, *_ in line_reqs:
            mentioned.update(sk)
        cleaned = []
        for sk, sp, items, min_hits in line_reqs:
            items = tuple(t for t in (tuple(x for x in it if x not in SOFT_SKILLS) for it in items) if t)
            sk = tuple(x for x in sk if x not in SOFT_SKILLS)
            if sk and items:
                cleaned.append((sk, sp, items, min(min_hits, len(items))))
        line_reqs = cleaned
        line_spans = [sp for _, sps, _, _ in line_reqs for sp in sps]
        for skills, spans, items, min_hits in line_reqs:
            importance = "normal"
            if section == "required":
                importance = "normal" if _LOW_BAR_RE.match(line) else "must"
            elif section == "preferred":
                importance = "nice"
            inline = _inline_importance(line, spans, line_spans)
            if inline:
                importance = inline
            key = tuple(sorted(skills)) + (("#examples",) if len(items) > 1 else ())
            if key in reqs:
                cur = reqs[key]
                rank = {"nice": 0, "normal": 1, "must": 2}
                if rank[importance] > rank[cur.importance]:
                    cur.importance = importance
            else:
                reqs[key] = Requirement(tuple(skills), importance, line.strip()[:200], items, min_hits)
                order.append(key)

    body = jd_text.replace(title, " ") if title else jd_text
    body = "\n".join(_DEGREE_SUBJECT_RE.sub(" ", _drop_team_clause(ln)) for ln in body.splitlines()
                     if not _is_heading_line(ln))
    discovered = extract_skills(body, infer=False)
    for extra in sorted(discovered - mentioned - SOFT_SKILLS):
        key = (extra,)
        if key not in reqs:
            reqs[key] = Requirement((extra,), "normal", "")
            order.append(key)

    rank = {"nice": 0, "normal": 1, "must": 2}
    final = []
    keys = [frozenset(k) for k in order]
    for k, fk in zip(order, keys):
        r = reqs[k]
        if "#examples" in fk:
            final.append(r)
            continue
        stricter = [order[i] for i, other in enumerate(keys) if other < fk and "#examples" not in other]
        if stricter:
            for sk in stricter:
                if rank[r.importance] > rank[reqs[sk].importance]:
                    reqs[sk].importance = r.importance
            continue
        final.append(r)

    musts = [r for r in final if r.importance == "must"]
    if final and len(musts) / len(final) > MUST_HAVE_MAX_SHARE:
        for r in musts:
            if not _INLINE_MUST_RE.search(r.source):
                r.importance = "normal"
    return final


@dataclass
class JDProfile:
    """Everything the scoring model needs to know about one JD."""
    text: str
    title: str = ""
    title_tokens: list = field(default_factory=list)
    seniority: str = "mid"
    required_years: float = 0.0
    prefers_freshers: bool = False
    fresher_role: bool = False
    graduates_only: bool = False
    bachelors_required: bool = False
    requirements: list = field(default_factory=list)
    skills: set = field(default_factory=set)
    title_skills: set = field(default_factory=set)
    education_fields: set = field(default_factory=set)
    warnings: list = field(default_factory=list)

    @property
    def effective_required_years(self) -> float:
        """Explicit minimum if stated, else the minimum implied by the seniority."""
        if self.required_years > 0:
            return self.required_years
        return SENIORITY_YEARS.get(self.seniority, 0.0)


_EDU_LINE_RE = re.compile(r"\b(?:bachelor|master|degree|graduate|qualification|discipline|b\.?s\.?c?|mba|bba|"
                          r"m\.?s\.?c?|phd|field\s+of\s+study|major(?:ing)?\s+in)\b", re.IGNORECASE)


def _education_lines(jd_text: str) -> str:
    """Only lines that talk about degrees - "strong communication skills"
    must not become a required field of study."""
    return "\n".join(ln for ln in jd_text.splitlines() if _EDU_LINE_RE.search(ln))


def build_profile(jd_text: str) -> JDProfile:
    """Resolve a JD into a requirement profile — call this once."""
    jd_text = jd_text or ""
    title = extract_job_title(jd_text)
    req_text = re.sub(r'\bFast(?=\s*(?:and\s*/\s*or|or|and|/|,)\s*Flask\b)', 'FastAPI', jd_text)
    requirements = extract_requirements(req_text, title)
    skills = {s for r in requirements for s in r.skills}
    seniority = detect_seniority(jd_text, title)

    warnings = []
    if not requirements:
        warnings.append("No recognisable skills found in this JD — scores will rely on "
                        "experience, education and title only.")
    elif len(requirements) < 3:
        warnings.append(f"Only {len(requirements)} recognisable skill requirement(s) found in this JD. "
                        "The skill vocabulary is strongest for software/AI/data roles; check the "
                        "interpretation below and treat scores as indicative.")
    if not title:
        warnings.append("No job title detected — title matching is disabled.")

    return JDProfile(
        text=jd_text,
        title=title,
        title_tokens=title_tokens(title),
        title_skills={sk for sk in (canonicalize_skill(c) for c, _, _ in skill_mentions(title))
                      if sk in skills and sk not in SOFT_SKILLS},
        seniority=seniority,
        required_years=extract_required_years(jd_text),
        prefers_freshers=bool(_FRESHER_PREFERRED_RE.search(jd_text)) or seniority == "entry",
        fresher_role=is_fresher_role(jd_text, seniority),
        graduates_only=requires_graduates_only(jd_text),
        bachelors_required=requires_bachelors(jd_text),
        requirements=requirements,
        skills=skills,
        education_fields=extract_education(_education_lines(jd_text)),
        warnings=warnings,
    )


__all__ = ["JDProfile", "Requirement", "build_profile", "canonicalize_skill", "ROLE_FAMILIES",
           "title_tokens", "extract_required_years", "detect_seniority", "extract_job_title"]

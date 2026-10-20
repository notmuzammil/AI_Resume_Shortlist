"""
Work-experience extraction.

Produces, per CV:
  * every dated ROLE (date range + the text block describing it),
  * TOTAL years  = union of all role intervals (overlaps merged),
  * the role blocks, so the pipeline can decide which roles are
    RELEVANT to a given JD and compute RELEVANT years separately.

Date handling: "Jan 2020 – Mar 2022", "Sept 2019 to Present",
"06/2021 - 08/2023", "01/02/2019 – 31/12/2021" (DD/MM/YYYY),
"2018 – 2020", pre-2000 years, and "Present/Current/Now/Till date".
Ranges are counted inclusively by month (Jun–Jul = 2 months) and never
past today.
"""
import re
from dataclasses import dataclass, field
from datetime import date

MONTHS_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7,
    'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


def _today():
    """Evaluated per call — a long-running server must not freeze 'Present'."""
    t = date.today()
    return t.year, t.month


_SEC_PATTERNS = [
    ('experience', r'(?:work\s+experiences?|professional\s+experiences?|employment(?:\s+history)?|career\s+history|'
                   r'professional\s+background|positions?\s+held|work\s+history|job\s+history|'
                   r'relevant\s+experience|industry\s+experience|internships?|'
                   r'e\s*x\s*p\s*e\s*r\s*i\s*e\s*n\s*c\s*e\s*s?|experience)'),
    ('education', r'(?:e\s*d\s*u\s*c\s*a\s*t\s*i\s*o\s*n|education(?:\s+(?:and|&)\s+(?:training|qualifications?))?|'
                  r'academic\s+background|academic\s+qualifications?|qualifications?|schooling)'),
    ('skills', r'(?:t\s*e\s*c\s*h\s*n\s*i\s*c\s*a\s*l\s+s\s*k\s*i\s*l\s*l\s*s|technical\s+skills?|additional\s+skills?|'
               r'key\s+strengths|core\s+competenc(?:y|ies)|areas\s+of\s+expertise|technical\s+expertise|'
               r'tools\s+(?:and|&)\s+technologies|'

               r'core\s+skills?|key\s+skills|s\s*k\s*i\s*l\s*l\s*s|skills|languages?\s+and\s+tools|tech\s+stack)'),
    ('projects', r'(?:p\s*r\s*o\s*j\s*e\s*c\s*t\s*s?|projects?|personal\s+projects?|academic\s+projects?|key\s+projects?|'
                 r'research\s+(?:and|&)\s+projects?|proects|final\s+year\s+project)'),
    ('certifications', r'(?:c\s*e\s*r\s*t\s*i\s*f\s*i\s*c\s*a\s*t\s*e\s*s?|certifications?|certificates?|courses?|'
                       r'certifications?\s+(?:and|&)\s+courses?|training)'),
    ('other', r'(?:leadership(?:\s+experience)?|extracurricular(?:\s+(?:and|&)\s+social\s+work)?|activities|'
              r'volunteer(?:ing)?|publications?|research|achievements?|awards?(?:\s+(?:and|&)\s+honou?rs)?|'
              r'honou?rs(?:\s+(?:and|&)\s+awards)?|languages?|l\s*a\s*n\s*g\s*u\s*a\s*g\s*e|references?|hobbies?|'
              r'interests?|objective|summary|profile|about\s+me|additional\s+information|'
              r'personal\s+information|presentations?|conferences?(?:\s+(?:and|&)\s+workshops)?|'
              r'community\s+service|recommendations)'),
]

_COMBINED_PAT = '|'.join(f'(?P<{name}>{pat})' for name, pat in _SEC_PATTERNS)
_HEADER_RE = re.compile(
    r'(?:^|[\n\r]|(?:\s{2,}))'
    r'[^\w\n]{0,3}'
    r'(?!(?:with|in|on|user|hands-on|real|mini|my|their|of|for|practical|academic|agile|key|all|some|more|and|'
    r'learning|following|using)\s+)'
    r'(?<![\(\[\/])'
    r'(?:' + _COMBINED_PAT + r')'
    r'(?:\s*[:\-\|]|\s*$|\s*[\n\r]|\s{2,})',
    re.IGNORECASE | re.MULTILINE
)
_CAPS_HEADER_RE = re.compile(
    r'(?<![A-Za-z])(WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT(?: HISTORY)?|EXPERIENCE|EDUCATION|'
    r'SKILLS|PROJECTS|CERTIFICATIONS|LANGUAGES|AWARDS|INTERESTS|HOBBIES|SUMMARY|PROFILE|REFERENCES)'
    r'(?![A-Za-z])')


_INLINE_HEADER_RE = re.compile(
    r'^[^\w\n]{0,3}(?:'
    r'(?P<caps>WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT HISTORY|EMPLOYMENT|EXPERIENCE|'
    r'EDUCATION(?: AND TRAINING| & TRAINING)?|ACADEMIC QUALIFICATIONS?|QUALIFICATIONS|'
    r'PROJECTS|CERTIFICATIONS|SKILLS|TECHNICAL SKILLS)'
    r'|(?P<title>Work Experience|Professional Experience|Employment History|Education(?: and Training)?|'
    r'Academic Qualifications?)(?=\s+[A-Z])'
    r')(?=\s)',
    re.MULTILINE)


def _classify(header: str) -> str:
    for name, pat in _SEC_PATTERNS:
        if re.fullmatch(pat, header.strip(), re.IGNORECASE):
            return name
    return 'other'


def _find_section_spans(text: str) -> list:
    """[(category, content_start, content_end)] for every section header."""
    markers = []
    for m in _HEADER_RE.finditer(text):
        cat = 'other'
        for name, _ in _SEC_PATTERNS:
            if m.group(name):
                cat = name
                break
        markers.append((cat, m.start(), m.end()))
    for m in _CAPS_HEADER_RE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[line_start:line_end if line_end != -1 else len(text)].strip()
        if len(line) > 45 or not re.fullmatch(r'[A-Z &/]+', line):
            continue
        if any(s <= m.start() < e for _, s, e in markers):
            continue
        markers.append((_classify(m.group(1)), m.start(), m.end()))
    for m in _INLINE_HEADER_RE.finditer(text):
        if any(s <= m.start() < e or abs(s - m.start()) < 4 for _, s, e in markers):
            continue
        markers.append((_classify(m.group("caps") or m.group("title")), m.start(), m.end()))
    markers.sort(key=lambda x: x[1])

    spans = []
    for i, (name, _hdr_start, hdr_end) in enumerate(markers):
        content_end = markers[i + 1][1] if i + 1 < len(markers) else len(text)
        spans.append((name, hdr_end, max(hdr_end, content_end)))
    return spans


def _extract_experience_section(text: str) -> str:
    chunks = [text[s:e] for name, s, e in _find_section_spans(text) if name == 'experience']
    return "\n".join(chunks)


def _strip_education_block(text: str) -> str:
    """
    Text with the Education section(s) removed.

    FIX: the old version rebuilt the text from the section spans only,
    silently dropping everything BEFORE the first detected heading. When
    the "Experience" heading itself wasn't detected (e.g. two-column
    "SKILLS EMPLOYMENT"), that preamble was the whole job history and
    the CV scored 0 years.
    """
    spans = _find_section_spans(text)
    if not spans:
        return text
    out = []
    pos = 0
    for name, start, end in spans:
        if name == 'education':
            out.append(text[pos:start])
            pos = end
    out.append(text[pos:])
    return "\n".join(out)


_MONTH_PAT = (r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|'
              r'sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?')
_PRESENT_PAT = r'(present|current(?:ly)?|now|today|till\s*(?:date|now)|to\s*date|ongoing|continuing)'
_YEAR_PAT = r'((?:19[89]|20[0-4])\d)'
_SEP = r'\s*(?:[-–—~]+|\bto\b|\btill\b|\buntil\b)\s*'

_RANGE_MONTH_YEAR_RE = re.compile(
    r'(?<![A-Za-z])' + _MONTH_PAT + r'[\s,\-/(]*' + _YEAR_PAT + r'\)?' + _SEP +
    r'(?:' + _MONTH_PAT + r'[\s,\-/(]*' + _YEAR_PAT + r'\)?|' + _PRESENT_PAT + r')',
    re.IGNORECASE)
_RANGE_YEAR_PAREN_MONTHS_RE = re.compile(
    r'(?<![\d/.\-])' + _YEAR_PAT + r'\s*\(\s*' + _MONTH_PAT + _SEP +
    r'(?:' + _MONTH_PAT + r'|' + _PRESENT_PAT + r')\s*\)',
    re.IGNORECASE)
_RANGE_MONTH_YEAR_TO_YEAR_RE = re.compile(
    r'(?<![A-Za-z])' + _MONTH_PAT + r'[\s,\-/]*' + _YEAR_PAT + _SEP + _YEAR_PAT + r'(?![\d/])',
    re.IGNORECASE)
_RANGE_YEAR_ONLY_RE = re.compile(
    r'(?<![\d/.\-])' + _YEAR_PAT + _SEP + r'(?:' + _YEAR_PAT + r'(?![\d/])|' + _PRESENT_PAT + r')',
    re.IGNORECASE)
_NUM_MONTH = r'(?<![\d/])(?:\d{1,2}[/.\-])?(0?[1-9]|1[0-2])[/.\-]'
_RANGE_NUMERIC_RE = re.compile(
    _NUM_MONTH + _YEAR_PAT + _SEP + r'(?:' + r'(?:\d{1,2}[/.\-])?(0?[1-9]|1[0-2])[/.\-]' + _YEAR_PAT +
    r'|' + _PRESENT_PAT + r')',
    re.IGNORECASE)


@dataclass
class Role:
    start: int
    end: int
    text: str
    header: str
    char_start: int = 0
    relevant: bool = False
    relevance_reason: str = ""

    @property
    def months(self) -> int:
        return max(0, self.end - self.start)

    def label(self) -> str:
        y1, m1 = divmod(self.start, 12)
        y2, m2 = divmod(self.end - 1, 12)
        return f"{m1 + 1:02d}/{y1} – {m2 + 1:02d}/{y2}"


@dataclass
class UndatedRole:
    """A job heading with no dates: reported, never counted in years."""
    header: str
    text: str
    relevant: bool = False
    relevance_reason: str = ""


@dataclass
class ExperienceResult:
    total_years: float = 0.0
    roles: list = field(default_factory=list)
    source: str = "none"
    excluded: list = field(default_factory=list)
    undated: list = field(default_factory=list)


def _abs(month: int, year: int) -> int:
    return year * 12 + (month - 1)


def _month_num(s) -> int:
    return MONTHS_MAP.get((s or '').lower()[:3], 1)


_EDU_CONTEXT_RE = re.compile(
    r'\b(?:bachelor|master|b\.?s\b|m\.?s\b|m\.?phil|ph\.?d|b\.?e\b|b\.?sc|m\.?sc|bscs|bsse|bsit|associate\s+(?:of|degree)|'
    r'degree|certificate|certification|certified|diploma|matric(?:ulation)?|intermediate|fsc|ics|hssc|ssc|'
    r'a[\s-]levels?|o[\s-]levels?|coursework|dissertation|thesis|gpa|cgpa|semester)\b',
    re.IGNORECASE)
_DEGREE_LINES_RE = re.compile(
    r"\b(?:bachelor'?s?|master'?s|associate'?s?\s+degree|degree|diploma|b\.?sc|m\.?sc|b\.\s?s\.?|"
    r"bs\s+(?:in|of|computer|software|information|electrical)|ph\.?d|m\.?phil|matric(?:ulation)?|"
    r"intermediate|fsc|hssc|a[\s-]levels?|o[\s-]levels?|bscs|bsse|bsit)\b",
    re.IGNORECASE)
_ACADEMIC_CONTEXT_RE = re.compile(
    r'\b(?:academic|personal\s+projects?|university\s+projects?|semester\s+projects?|course\s*work|'
    r'fyp|final\s+year\s+project|side\s+projects?|hobby\s+projects?|self[\s-]?(?:learning|study|taught)|'
    r'student\s+(?:society|club|chapter)|society|volunteer(?:ing)?)\b',
    re.IGNORECASE)


def _line_bounds(text: str, pos: int):
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return s, (len(text) if e == -1 else e)


def _context_lines(text: str, start: int, end: int) -> str:
    """The line holding the range plus the line before it (title/company)."""
    ls, le = _line_bounds(text, start)
    prev_s = text.rfind("\n", 0, max(0, ls - 1)) + 1 if ls > 0 else ls
    return text[prev_s:le]


_PLACEHOLDER_RE = re.compile(
    r'(?<![\w/.])0{1,2}\s*[,/.\-]\s*0{1,4}(?![\w/.])|(?-i:\bYYYY\b|\b(?:DD\s*[/\-]\s*)?MM\s*[/\-]\s*YY(?:YY)?\b)'
    r'|\b(?:19|20)XX\b'
    r'|\bmonth\s*,?\s*year\b|\b(?:start|end)\s+date\b|\bdate\s*[-–—]\s*date\b',
    re.IGNORECASE)


def _collect_ranges(section: str, is_experience_section: bool):
    """[(abs_start, abs_end_exclusive, char_start, char_end)], plus exclusions."""
    now_y, now_m = _today()
    now_abs = _abs(now_m, now_y) + 1
    taken = []
    out = []
    excluded = []

    def overlaps(s, e):
        return any(not (e <= ts or s >= te) for ts, te in taken)

    def add(abs_s, abs_e, m):
        if overlaps(m.start(), m.end()):
            return
        ctx_line = section[_line_bounds(section, m.start())[0]:_line_bounds(section, m.start())[1]]
        if _PLACEHOLDER_RE.search(ctx_line):
            excluded.append((ctx_line.strip()[:60], "template placeholder date, not a real job"))
            taken.append((m.start(), m.end()))
            return
        ctx = _context_lines(section, m.start(), m.end())
        ls0 = _line_bounds(section, m.start())[0]
        p1 = section.rfind("\n", 0, max(0, ls0 - 1)) + 1 if ls0 > 0 else ls0
        p2 = section.rfind("\n", 0, max(0, p1 - 1)) + 1 if p1 > 0 else p1
        above2 = section[p2:ls0]
        if (_EDU_CONTEXT_RE.search(ctx_line) or (_EDU_CONTEXT_RE.search(ctx) and not is_experience_section)
                or _DEGREE_LINES_RE.search(above2)
                or (not is_experience_section and re.search(r'\b(?:university|college|institute|school)\b',
                                                            ctx, re.IGNORECASE))):
            excluded.append((m.group(0), "education/certificate dates"))
            taken.append((m.start(), m.end()))
            return
        if _ACADEMIC_CONTEXT_RE.search(ctx):
            excluded.append((m.group(0), "academic/personal project, not employment"))
            taken.append((m.start(), m.end()))
            return
        abs_e = min(abs_e, now_abs)
        if abs_s >= now_abs or abs_e - abs_s <= 0 or abs_e - abs_s > 12 * 45:
            return
        taken.append((m.start(), m.end()))
        out.append((abs_s, abs_e, m.start(), m.end()))

    for m in _RANGE_MONTH_YEAR_RE.finditer(section):
        sm, sy, em, ey, present = m.groups()
        start = _abs(_month_num(sm), int(sy))
        end = now_abs if present else _abs(_month_num(em), int(ey)) + 1
        add(start, end, m)
    for m in _RANGE_YEAR_PAREN_MONTHS_RE.finditer(section):
        y, sm, em, present = m.groups()
        start = _abs(_month_num(sm), int(y))
        if present:
            end = now_abs
        else:
            end_m = _month_num(em)
            end = _abs(end_m, int(y) + (1 if end_m < _month_num(sm) else 0)) + 1
        add(start, end, m)
    for m in _RANGE_NUMERIC_RE.finditer(section):
        sm, sy, em, ey, present = m.groups()
        start = _abs(int(sm), int(sy))
        end = now_abs if present else _abs(int(em), int(ey)) + 1
        add(start, end, m)
    for m in _RANGE_MONTH_YEAR_TO_YEAR_RE.finditer(section):
        sm, sy, ey = m.groups()
        add(_abs(_month_num(sm), int(sy)), _abs(12, int(ey)) + 1, m)
    for m in _RANGE_YEAR_ONLY_RE.finditer(section):
        sy, ey, present = m.groups()
        start = _abs(6, int(sy))
        end = now_abs if present else _abs(6, int(ey))
        add(start, end, m)
    out.sort(key=lambda r: r[2])
    return out, excluded


_ROLE_LINE_RE = re.compile(
    r'\b(?:engineer|developer|programmer|scientist|analyst|designer|manager|architect|administrator|'
    r'consultant|specialist|intern|trainee|officer|executive|lead|researcher|tester|instructor|teacher|'
    r'lecturer|agent|associate|assistant|coordinator|generalist|recruiter|accountant|representative|'
    r'supervisor|director|advisor)s?\b', re.IGNORECASE)


_ICON_GLYPHS_RE = re.compile(r'[\ue000-\uf8ff]')
_BULLET_START_RE = re.compile(r'^\s*(?:[•●▪◦‣\-*–·>]|\d{1,2}[.)])')
_DESC_VERB_RE = re.compile(
    r'^(?:\w+ed|working|developing|building|managing|leading|handling|creating|designing|implementing|using|'
    r'collaborating|maintaining|providing|supporting|assisting|teaching|responsible|i|my|we|our|the|a|an|'
    r'and|with|in|on|for|to|of|tools?|tech(?:nologies)?|skills?|environment|stack)\b', re.IGNORECASE)


def _is_role_heading(line: str) -> bool:
    """A short heading naming a job: "Flutter developer", "Intern at Gexton –
    Hyderabad, Pakistan", "Teacher (computer science)". Not a description
    line ("Led a team of 5 developers", "• Worked as a QA engineer")."""
    s = _ICON_GLYPHS_RE.sub('', line).strip()
    words = s.split()
    if not words:
        return False
    if not (1 <= len(words) <= 8) or s.endswith(('.', ',', ';', ':')) or ':' in s:
        return False
    if _BULLET_START_RE.match(s) or not s[0].isupper() or _DESC_VERB_RE.match(s):
        return False
    if not (_ROLE_LINE_RE.search(s) or re.search(r'\bat\s+[A-Z]', s)):
        return False
    caps = sum(1 for w in words if w[0].isupper() or not w[0].isalpha())
    return len(words) <= 3 or caps / len(words) >= 0.5


def _split_undated(text: str, start: int, end: int):
    """Positions in text[start:end] (the part of a dated role AFTER its date
    line) where an undated job begins: a role heading that follows at least
    one line of description. Two heading lines in a row ("Intern – Hyderabad"
    then "Flutter developer") start one job."""
    cuts = []
    seen_desc = False
    prev_heading = False
    pos = start
    while pos < end:
        le = text.find("\n", pos)
        le = end if le == -1 or le > end else le
        line = text[pos:le]
        if line.strip():
            heading = _is_role_heading(line)
            if heading and seen_desc and not prev_heading:
                cuts.append(pos)
                seen_desc = False
            elif not heading:
                seen_desc = True
            prev_heading = heading
        pos = le + 1
    return cuts


def _merge_months(intervals) -> int:
    if not intervals:
        return 0
    iv = sorted(intervals)
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def merged_years(roles) -> float:
    return round(_merge_months([(r.start, r.end) for r in roles]) / 12, 1)


_STATED_YEARS_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s*)?(?:industry\s+|professional\s+|work\s+|total\s+|hands[\s-]on\s+)?'
    r'(?:experience|exp)\b', re.IGNORECASE)


def analyse_experience(text: str) -> ExperienceResult:
    """Dated roles + total years (union of intervals)."""
    exp_section = _extract_experience_section(text)
    in_exp = bool(exp_section.strip())
    search_text = exp_section if in_exp else _strip_education_block(text)

    ranges, excluded = _collect_ranges(search_text, in_exp)

    def role_start(cs, ce):
        """Line holding the date — extended up to the line above when the
        date stands alone, or when that line is a short job title
        ("Jr. Front-End Developer" above "icreativez | March 2025 – Nov 2025")."""
        ls, le = _line_bounds(search_text, cs)
        if ls == 0:
            return ls
        prev_start = search_text.rfind("\n", 0, ls - 1) + 1
        prev = search_text[prev_start:ls].strip()
        rest = (search_text[ls:cs] + search_text[ce:le])
        if len(re.findall(r'[A-Za-z]{3,}', rest)) < 2:
            return prev_start
        if prev and len(prev.split()) <= 8 and _ROLE_LINE_RE.search(prev) \
                and not _RANGE_MONTH_YEAR_RE.search(prev) and not _RANGE_YEAR_ONLY_RE.search(prev):
            return prev_start
        return ls

    def with_company_line(start):
        """Also take a company heading above the title ("Intern at Gexton –
        Hyderabad" over "Java developer" over the date) — it belongs to this
        job, not to the end of the previous one."""
        ls = _line_bounds(search_text, start)[0]
        if ls == 0:
            return ls
        prev_start = search_text.rfind("\n", 0, ls - 1) + 1
        prev = search_text[prev_start:ls]
        if _is_role_heading(prev) and not _RANGE_MONTH_YEAR_RE.search(prev) \
                and not _RANGE_YEAR_ONLY_RE.search(prev) and not _RANGE_NUMERIC_RE.search(prev):
            return prev_start
        return ls

    starts = []
    for _, _, cs, ce in ranges:
        st = role_start(cs, ce)
        starts.append(with_company_line(st) if st < _line_bounds(search_text, cs)[0] else st)

    cuts_by_role = []
    for i, (_, _, cs, ce) in enumerate(ranges):
        date_line_end = _line_bounds(search_text, ce)[1]
        block_end = max(date_line_end, starts[i + 1] if i + 1 < len(ranges) else len(search_text))
        cuts = _split_undated(search_text, date_line_end, block_end)
        if cuts and i + 1 < len(ranges):
            nxt_head = search_text[starts[i + 1]:ranges[i + 1][2]]
            if search_text.count("\n", cuts[-1], ranges[i + 1][2]) <= 6 \
                    and not any(_is_role_heading(ln) for ln in nxt_head.splitlines()):
                starts[i + 1] = min(starts[i + 1], cuts.pop())
        cuts_by_role.append(cuts)

    roles, undated = [], []
    for i, (s, e, cs, ce) in enumerate(ranges):
        block_start = max(starts[i], ranges[i - 1][3] if i else 0)
        date_line_end = _line_bounds(search_text, ce)[1]
        block_end = max(date_line_end, starts[i + 1] if i + 1 < len(ranges) else len(search_text))
        cuts = cuts_by_role[i]
        for a, b in zip(cuts, cuts[1:] + [block_end]):
            chunk = search_text[a:b].strip()
            head = " / ".join(_ICON_GLYPHS_RE.sub('', ln).strip()
                              for ln in chunk.splitlines()[:2] if _is_role_heading(ln))
            undated.append(UndatedRole(head or chunk.splitlines()[0][:80], chunk))
        if cuts:
            block_end = cuts[0]
        block = search_text[block_start:block_end]
        header = _ICON_GLYPHS_RE.sub('', search_text[block_start:date_line_end])
        roles.append(Role(s, e, block.strip(), header.strip(), cs))

    if roles:
        return ExperienceResult(merged_years(roles), roles, "dated-roles", excluded, undated)

    for scope in ((exp_section,) if in_exp else ()) + (text,):
        found = [float(y) for y in _STATED_YEARS_RE.findall(scope) if 0 < float(y) <= 45]
        if found:
            return ExperienceResult(round(max(found), 1), [], "stated-years", excluded)
    return ExperienceResult(0.0, [], "none", excluded)


_ROLE_NOUNS = (
    r'engineer|developer|programmer|scientist|analyst|designer|'
    r'manager|architect|administrator|consultant|specialist|'
    r'intern|trainee|officer|executive|lead|researcher|tester|instructor|teacher|lecturer|agent|'
    r'generalist|recruiter|coordinator|accountant|representative|supervisor|director|advisor|'
    r'assistant|associate(?!\s+(?:degree|of\b|in\b))|business\s+partner'
)

_CANDIDATE_TITLE_RE = re.compile(
    r'\b((?-i:[A-Z])[A-Za-z+/#.\-]*(?:[ ][A-Z][A-Za-z+/#.\-&]*){0,3}[ ]'
    r'(?:' + _ROLE_NOUNS + r')s?)\b',
    re.IGNORECASE
)


def extract_job_titles(text: str) -> list:
    """
    Job titles the candidate has held, scoped to the Experience section
    (falls back to the education-stripped text).
    """
    section = _extract_experience_section(text)
    if not section.strip():
        section = _strip_education_block(text)

    spans = _find_section_spans(text)
    head_end = min([s for _, s, _ in spans] + [len(text)])
    headline = "\n".join(text[:head_end].splitlines()[:6])

    titles = []
    seen = set()
    for m in list(_CANDIDATE_TITLE_RE.finditer(headline)) + list(_CANDIDATE_TITLE_RE.finditer(section)):
        t = re.sub(r'\s+', ' ', m.group(1)).strip()
        if re.match(r'(?i)(?:with|and|as|the|a|an|for|of|to|in|at)\b', t):
            t = t.split(" ", 1)[1] if " " in t else t
        key = t.lower()
        if key not in seen and len(t) <= 60:
            seen.add(key)
            titles.append(t)
    return titles[:12]

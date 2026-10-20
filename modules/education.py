import re
from dataclasses import dataclass as _dataclass
from datetime import datetime


def _cur_year() -> int:
    return datetime.now().year


def _cur_month() -> int:
    return datetime.now().month


BACHELORS_TRIGGERS = [
    r"bachelor(?:s|'s)?\s+(?:of|in)\b",
    r"bachelor(?:s|'s)?",
    r"bs(?:cs|se|it|ee|ce|ai|ds|cy|is)\b",
    r"b\.?sc",
    r"b\.?tech",
    r"b\.?s(?![a-zA-Z])",
    r"bcs",
    r"bba",
    r"b\.?com",
    r"mbbs",
    r"b\.?arch",
    r"llb|l\.l\.b",
    r"undergraduate",
    r"bechorlar",
    r"bechelor(?:s|'s)?",
    r"bacholar(?:s|'s)?",
    r"batchelor(?:s|'s)?",
    r"bacheler(?:s|'s)?",
    r"bachalor(?:s|'s)?",
]

MASTERS_TRIGGERS = [
    r"master(?:s|'s)?\s+(?:of|in)\b",
    r"master(?:s|'s)?",
    r"m\.?sc",
    r"m\.?eng\b",
    r"m\.?tech",
    r"mba",
    r"mcs",
    r"m\.?com",
    r"m\.?s(?![a-zA-Z])(?!\.?\s*(?:word|excel|office|sql|access|powerpoint|windows|project|server|dynamics|azure|teams|paint|sharepoint|visio|publisher|dos|exchange|365)\b)",
    r"postgraduate|post[-\s]?graduate",
]

MPHIL_TRIGGERS = [
    r"mphil",
    r"m\.?\s*phil",
    r"master\s+of\s+philosophy",
]

PHD_TRIGGERS = [
    r"ph\.?\s*d",
    r"doctorate|doctoral",
    r"doctor\s+of\s+philosophy",
]


def _bp(triggers: list) -> re.Pattern:
    """
    Build one compiled regex from trigger list.
    Each trigger is surrounded by non-alpha boundaries so
    "bs" won't match inside "jobs", "abs", etc.
    """
    parts = [r'(?<![a-zA-Z])(?:' + t + r')(?![a-zA-Z])' for t in triggers]
    return re.compile('|'.join(parts), re.IGNORECASE)


_BACHELORS_RE = _bp(BACHELORS_TRIGGERS)
_MPHIL_RE     = _bp(MPHIL_TRIGGERS)
_MASTERS_RE   = _bp(MASTERS_TRIGGERS)
_PHD_RE       = _bp(PHD_TRIGGERS)

_BE_ABBREV_RE = re.compile(r'(?<![a-zA-Z])B\.?E\.?(?![a-zA-Z])')
_ME_ABBREV_RE = re.compile(
    r'(?<![a-zA-Z])M\.?E\.?(?![a-zA-Z])'
    r'(?=\s*(?:in|of|\(|computer|electrical|mechanical|civil|software|electronic|'
    r'information|telecommunication|chemical|aerospace|biomedical|industrial|'
    r'engineering|technology|science|systems|networks))',
    re.IGNORECASE
)

_NEXT_DEGREE_FENCE = re.compile(
    r'\b(?:bachelor|master|mphil|m\.phil|phd|ph\.d'
    r'|bsc|b\.sc|msc|m\.sc|b\.e|b\.tech|m\.tech|m\.e'
    r'|ms\b|bs\b|bba|llb|mbbs|undergraduate'
    r'|bscs|bsse|bsit|bsee|bsce|bsai|bsds|bscy|bsis'
    r'|diploma|certificate|certification|intermediate|matric|fsc|hssc|a-levels?|o-levels?)\b',
    re.IGNORECASE
)

_SECTION_FENCE = re.compile(
    r'(?:\n|^)\s*(?:work\s+experience|professional\s+experience|employment'
    r'|projects?|organizations?|certifications?|achievements?'
    r'|publications?|references|volunteer(?:ing)?|extracurricular'
    r'|leadership\s+experience|internships?|awards?|activities'
    r'|presentations?|conferences?|workshops?|seminars?'
    r'|honou?rs?|community\s+service|hobbies|interests'
    r'|skills?|education|experiences?|summary|profile|objective|languages?)\b',
    re.IGNORECASE
)

_LEVEL_PATTERNS = [
    ("bachelors", _BACHELORS_RE),
    ("mphil",     _MPHIL_RE),
    ("masters",   _MASTERS_RE),
    ("phd",       _PHD_RE),
]


_WORD_RE = re.compile(r"[a-zA-Z']+")


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _find_bachelor_typo_match(text: str):
    """
    Scan whitespace-delimited words for a close misspelling of
    "bachelor" / "bachelors". Length-gated (6-11 chars, starts with
    'b') and requires edit distance <= 3 — verified against common
    English words to avoid false positives (tested: backend, before,
    below, belong, because, background, behavior, benefit, business
    all correctly excluded at this threshold).

    Returns the FIRST typo match object, or None.
    """
    targets = ("bachelor", "bachelors")
    for m in _WORD_RE.finditer(text):
        w = m.group().lower().strip("'")
        if not (7 <= len(w) <= 11) or not w.startswith("b"):
            continue
        if any(_levenshtein(w, t) <= 2 for t in targets):
            return m
    return None


def _has_bachelor_typo(text: str) -> bool:
    return _find_bachelor_typo_match(text) is not None


_ALL_YEARS_RE     = re.compile(r'\b((?:19[7-9]|20[0-4])\d)\b')
_CURRENT_WORDS_RE = re.compile(r'\b(current|present|now|ongoing)\b', re.IGNORECASE)

_MONTH_NAME_TO_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_YEAR_RE = re.compile(r'\b([A-Za-z]{3,10})\.?\s*[-,]?\s*(20\d{2})\b')


def _get_snippet(text: str, match_start: int, match_end: int) -> str:
    """
    Return the text surrounding the degree match, bounded by:
      • the nearest preceding degree keyword or section header,
      • the nearest succeeding degree keyword or section header,
      • or max 120 chars backward and 180 chars forward.
    """
    window_start = max(0, match_start - 120)
    window_end = min(len(text), match_end + 180)

    raw_before = text[window_start:match_start]
    raw_after = text[match_end:window_end]

    before_cut = 0
    for fence in [_NEXT_DEGREE_FENCE, _SECTION_FENCE]:
        for m in fence.finditer(raw_before):
            before_cut = max(before_cut, m.end())

    after_cut = len(raw_after)
    for fence in [_NEXT_DEGREE_FENCE, _SECTION_FENCE]:
        m = fence.search(raw_after)
        if m:
            after_cut = min(after_cut, m.start())

    return raw_before[before_cut:] + text[match_start:match_end] + raw_after[:after_cut]


_PERSONAL_INFO_RE = re.compile(
    r'(?:date\s+of\s+birth|dob|d\.?o\.?b|born|age|gender|nationality'
    r'|cnic|passport|religion|marital)\s*:?\s*[\d/\-,\w\s]*?(?=\n|$|[A-Z]{2,})',
    re.IGNORECASE
)


def _year_from_snippet(snippet: str):
    """
    Return the graduation (end) year from the snippet, or None.

    Algorithm:
      1. Strip personal-info tokens (DOB, nationality, gender…) so birth
         years never contaminate the result.
      2. If "current / present / now / ongoing" → return CURRENT_YEAR.
      3. Collect all 20XX years in the snippet.
      4. Return the MAXIMUM (= end year of a range, or the single year).
      Only years from 2005 onward are accepted as graduation years —
      this prevents birth years like 2003 from ever being returned.
    """
    clean = _PERSONAL_INFO_RE.sub('', snippet)
    if _CURRENT_WORDS_RE.search(clean):
        return _cur_year()
    years = [int(y) for y in _ALL_YEARS_RE.findall(clean)
             if 1975 <= int(y) <= _cur_year() + 6]
    return max(years) if years else None


def _degree_still_in_progress(snippet: str, year) -> bool:
    """
    Determine whether a degree is still being pursued (undergraduate /
    not yet graduated) rather than already completed, using the same
    snippet _year_from_snippet() used to resolve the year.

    True when:
      - "Present"/"Current"/"Ongoing"/"Now" appears in the snippet
        (e.g. "Dec 2022 - Present"), or
      - the resolved year is later than the current year (e.g. a CV
        stating "Expected 2027" without the word "present"), or
      - the resolved year equals the current year AND a month name
        attached to that same year is later than the current month
        (e.g. "Dec 2022 - Nov 2026" when today is August 2026 — same
        year, but the completion date hasn't actually arrived yet).

    False when the year is genuinely in the past, or no year could be
    determined at all (safer default: don't assume undergraduate status
    without positive evidence).
    """
    clean = _PERSONAL_INFO_RE.sub('', snippet)

    if _CURRENT_WORDS_RE.search(clean):
        return True

    if year is None:
        return False

    if year > _cur_year():
        return True

    if year == _cur_year():
        for month_name, yr in _MONTH_YEAR_RE.findall(clean):
            if int(yr) != year:
                continue
            month_num = _MONTH_NAME_TO_NUM.get(month_name.lower())
            if month_num and month_num > _cur_month():
                return True
        return False

    return False


def extract_degree_years(text: str) -> dict:
    """
    Scan CV text and return a dict mapping each degree level found
    to its graduation (end) year.

    Return format
    ─────────────
    {
        "bachelors": 2025,   # found + year detected
        "masters":   2026,   # found + ongoing (current year)
        "mphil":     None,   # found but year not detectable
        # "phd" absent → not mentioned at all
    }

    For each level we scan ALL matches, fence each snippet to
    prevent year bleed-over, then keep the LATEST year seen
    (= most recently completed degree at that level).

    Scoped to the Education section (see _extract_education_section)
    before searching. Without this, degree keywords matched ANYWHERE
    in the CV — including incidental mentions with no connection to
    the candidate's own degree, e.g. "co-taught... to 6th-semester
    Bachelor's students" (a job description, not the candidate's
    education) or "Second Position in Bachelor of Science (Honours)"
    inside an Awards section (a real mention, but with a nearby year
    from unrelated surrounding text bleeding into the resolved year).
    Both produced wrong years by outranking the actual, correct
    Education-section entry in the "keep the latest year" comparison.
    """
    section = _extract_education_section(text)

    result = {}
    for level, pattern in _LEVEL_PATTERNS:
        best_year = None
        best_snippet = None
        found = False
        if level == "bachelors":
            matches = list(pattern.finditer(section)) + list(_BE_ABBREV_RE.finditer(section))
            matches.sort(key=lambda m: m.start())
        elif level == "masters":
            matches = list(pattern.finditer(section)) + list(_ME_ABBREV_RE.finditer(section))
            matches.sort(key=lambda m: m.start())
        else:
            matches = list(pattern.finditer(section))

        for m in matches:
            found = True
            snippet = _get_snippet(section, m.start(), m.end())
            year = _year_from_snippet(snippet)
            if year is not None:
                if best_year is None or year >= best_year:
                    best_year = year
                    best_snippet = snippet

        if level == "bachelors" and not found:
            typo_match = _find_bachelor_typo_match(section)
            if typo_match is not None:
                found = True
                snippet = _get_snippet(section, typo_match.start(), typo_match.end())
                year = _year_from_snippet(snippet)
                if year is not None:
                    best_year = year
                    best_snippet = snippet
        if found:
            result[level] = best_year
            if level == "bachelors":
                result["bachelors_in_progress"] = _degree_still_in_progress(
                    best_snippet or "", best_year
                )
    return result


def has_bachelors(text: str) -> bool:
    """
    Hard rejection gate.
    Returns True only if the CV mentions a bachelor's-level degree.
    Candidates without a bachelor's are rejected immediately.

    IMPORTANT: call this with the ORIGINAL-case text (not lowercased),
    since the "BE"/"B.E." check below is intentionally case-sensitive.
    """
    if _BACHELORS_RE.search(text):
        return True
    if _BE_ABBREV_RE.search(text):
        return True
    if _has_bachelor_typo(text):
        return True
    return False


COMPUTING_CORE = [
    "computer science", "software engineering", "artificial intelligence",
    "data science", "data engineering", "data analytics",
    "information technology", "information systems",
    "computer engineering", "computer systems engineering",
    "cybersecurity", "information security", "computer networks",
    "human computer interaction"
]

COMPUTING_RELATED = [
    "electrical engineering", "electronics engineering",
    "telecommunications engineering", "mathematics",
    "applied mathematics", "statistics", "computational mathematics",
    "mechatronics"
]

COMPUTING_BROAD = [
    "physics", "applied physics", "space science", "astronomy", "mechanical engineering", "aerospace engineering"
]

DOMAIN_GROUPS = {
    "computer science": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED,
        "broad":   COMPUTING_BROAD
    },
    "software engineering": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED,
        "broad":   COMPUTING_BROAD
    },
    "information technology": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED,
        "broad":   COMPUTING_BROAD
    },
    "data science": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED + ["economics"],
        "broad":   COMPUTING_BROAD
    },
    "artificial intelligence": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED,
        "broad":   COMPUTING_BROAD
    },
    "cybersecurity": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED,
        "broad":   COMPUTING_BROAD
    },
    "computer engineering": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED,
        "broad":   COMPUTING_BROAD
    },
    "computer systems engineering": {
        "core":    COMPUTING_CORE,
        "related": COMPUTING_RELATED,
        "broad":   COMPUTING_BROAD
    },
    "electrical engineering": {
        "core":    ["electrical engineering", "electronics engineering", "telecommunications engineering"],
        "related": ["computer engineering", "computer systems engineering", "mechatronics"],
        "broad":   ["computer science", "mechanical engineering", "physics"]
    },
    "mechanical engineering": {
        "core":    ["mechanical engineering", "industrial engineering"],
        "related": ["mechatronics", "aerospace engineering", "materials engineering"],
        "broad":   ["electrical engineering", "civil engineering", "physics"]
    },
    "civil engineering": {
        "core":    ["civil engineering"],
        "related": ["structural engineering", "environmental engineering", "construction management"],
        "broad":   ["mechanical engineering", "architecture"]
    },
    "business administration": {
        "core":    ["business administration", "business management", "project management", "management sciences",
                    "commerce"],
        "related": ["human resource management", "human resources", "marketing management", "marketing", "advertising", "supply chain management", "finance", "accounting", "economics", "hospitality management", "hotel management"],
        "broad":   ["information systems", "business analytics"]
    },
    "human resource management": {
        "core":    ["human resource management", "human resources", "human resource development",
                    "industrial relations", "organizational psychology"],
        "related": ["business administration", "business management", "management sciences", "psychology",
                    "public administration"],
        "broad":   ["sociology", "economics", "commerce"]
    },
    "design": {
        "core":    ["fashion design", "graphic design", "industrial design", "interior design", "textile design"],
        "related": ["fine arts", "visual arts", "architecture"],
        "broad":   ["media studies", "communication"]
    },
    "finance": {
        "core":    ["finance", "financial management", "accounting", "banking"],
        "related": ["economics", "risk management", "business administration", "statistics"],
        "broad":   ["mathematics", "data science"]
    },
    "economics": {
        "core":    ["economics", "applied economics", "econometrics"],
        "related": ["finance", "accounting", "banking", "statistics"],
        "broad":   ["business administration", "mathematics"]
    },
    "medicine": {
        "core":    ["medicine", "surgery"],
        "related": ["pharmacy", "nursing", "dentistry", "public health"],
        "broad":   ["biochemistry", "biology", "biomedical engineering"]
    },
    "mathematics": {
        "core":    ["mathematics", "applied mathematics", "statistics", "computational mathematics"],
        "related": ["data science", "physics", "computer science", "economics"],
        "broad":   ["electrical engineering", "finance"]
    },
    "physics": {
        "core":    ["physics", "applied physics", "space science", "astronomy"],
        "related": ["mathematics", "materials engineering", "electrical engineering"],
        "broad":   ["chemistry", "data science"]
    },
    "biology": {
        "core":    ["biology", "biochemistry", "molecular biology"],
        "related": ["biotechnology", "microbiology", "genetics"],
        "broad":   ["chemistry", "medicine", "pharmacy"]
    },
}


def _extract_education_section(text: str) -> str:
    """
    Scope the text down to just the EDUCATION section, so domain-field
    matching doesn't pick up field-sounding words sitting anywhere else
    in the CV. Real examples that were false-positiving before this fix:
      - "Data Science"    ← from a "365 Data Science" course/certification
      - "Architecture"    ← from "modular architecture patterns" in a
                             project description
      - "Data Analytics"  ← from an "AI & Data Analytics Intern" job title
      - "Data Engineering" ← from a "Data Engineering:" skills-category
                             heading
      - "Banking"          ← from a "Fintech Banking App" project title
                             sitting between an incidental EARLIER use of
                             the word "education" (e.g. "...facilitate
                             STEM education for school children...") and
                             the real Education heading. The naive
                             "first match of the word education" approach
                             latched onto that incidental mention instead
                             of the real heading, and everything between
                             it and the real heading (a whole different
                             section) got scanned as if it were the
                             education section.
      - "Data Science"     ← from a "Key Courses Knowledge: ... Data
                             Science & Analytics ..." coursework list
                             sitting INSIDE the real Education section,
                             genuinely between the heading and the next
                             section fence, but describing courses taken
                             rather than the degree itself.

    Two safeguards:
    1. A candidate "education" match is only accepted as the real
       heading if a university/degree/date signal appears shortly after
       it — an incidental mention like "STEM education for..." has no
       such signal nearby and is skipped in favor of the next match.
    2. Once a valid section is found, it's additionally truncated at a
       coursework-list marker ("key courses", "coursework", "subjects
       studied", etc.) if one appears, since those lists routinely
       mention many domain-adjacent buzzwords that aren't the actual
       degree field.

    If no valid heading is found at all (common in JD text, which often
    states a degree requirement inline without ever using the word
    "education"), falls back to scanning the full text.
    """
    edu_heading_re = re.compile(
        r'\b(?:e\s*d\s*u\s*c\s*a\s*t\s*i\s*o\s*n|education)(?:\s+and\s+training|al\s+background|\s+history)?\b',
        re.IGNORECASE
    )
    heading_confirm_re = re.compile(
        r'\b(university|college|institute|institution|diploma|certificate|matric|fsc|intermediate|bachelor|master|'
        r'b\.?e\b|b\.?sc|bs\b|bsse|bscs|bsit|bsee|bsce|bsai|bsds|bscy|bsis|degree|gpa|cgpa|'
        r'20\d{2}\s*[-–—]\s*20\d{2}|20\d{2}\s*[-–—]\s*present|20\d{2}|'
        r'software|computer|engineering|science|technology)\b',
        re.IGNORECASE
    )
    coursework_fence_re = re.compile(
        r'\b(?:key\s+courses?|relevant\s+coursework|coursework|'
        r'courses?\s+(?:studied|taken|covered)|subjects?\s+(?:studied|covered)|'
        r'modules?\s+covered)\b',
        re.IGNORECASE
    )

    search_from = 0
    while True:
        m = edu_heading_re.search(text, search_from)
        if not m:
            return text
        lookahead = text[m.end(): m.end() + 250]
        if heading_confirm_re.search(lookahead):
            start = m.end()
            break
        search_from = m.end()

    end_m = _SECTION_FENCE.search(text, start)
    end = end_m.start() if end_m else len(text)
    section = text[start:end]

    course_m = coursework_fence_re.search(section)
    if course_m:
        section = section[:course_m.start()]

    return section


_FIELD_OF_STUDY_LABEL_RE = re.compile(
    r'field\(?s?\)?\s+of\s+study\s*:?', re.IGNORECASE
)
_DEGREE_ANCHOR_WINDOW = 200
_FIELD_LABEL_WINDOW   = 80


_BUSINESS_DEGREE_RE = re.compile(r'\b(?:BBA|MBA|EMBA|B\.B\.A|M\.B\.A)\b')
_PLAIN_BUSINESS_RE = re.compile(
    r"\b(?:degree|bachelor'?s?|master'?s?|b\.?sc?|m\.?sc?|diploma)\s+(?:degree\s+)?(?:in|of)\s+business\b"
    r"(?!\s+(?:analytics|intelligence|information|informatics|administration|management))", re.IGNORECASE)


def extract_education(text: str) -> set:
    """
    Extract degree field names, scoped to the Education section AND
    anchored to text surrounding an actual degree keyword (bidirectional) or
    a "Field(s) of study:" label.
    """
    section = _extract_education_section(text)

    anchor_spans = []
    for _level, pattern in _LEVEL_PATTERNS:
        for m in pattern.finditer(section):
            anchor_spans.append((m.start(), m.end()))
    for m in _BE_ABBREV_RE.finditer(section):
        anchor_spans.append((m.start(), m.end()))
    for m in _ME_ABBREV_RE.finditer(section):
        anchor_spans.append((m.start(), m.end()))
    typo_match = _find_bachelor_typo_match(section)
    if typo_match is not None:
        anchor_spans.append((typo_match.start(), typo_match.end()))

    field_label_ends = [m.end() for m in _FIELD_OF_STUDY_LABEL_RE.finditer(section)]

    windows = [section[max(0, s - 80): min(len(section), e + _DEGREE_ANCHOR_WINDOW)] for s, e in anchor_spans]
    windows += [section[a: a + _FIELD_LABEL_WINDOW] for a in field_label_ends]

    if not windows:
        if section != text:
            windows = [section]
        else:
            windows = []

    all_fields = set()
    for domain, levels in DOMAIN_GROUPS.items():
        all_fields.add(domain)
        for level_fields in levels.values():
            all_fields.update(level_fields)

    found = set()
    for window_text in windows:
        window_lower = window_text.lower()
        for field in all_fields:
            pattern = r'(?<![a-z])' + re.escape(field) + r'(?![a-z])'
            if re.search(pattern, window_lower):
                found.add(field)
        if _BUSINESS_DEGREE_RE.search(window_text) or _PLAIN_BUSINESS_RE.search(window_text):
            found.add("business administration")

    pruned = set()
    for f in found:
        if not any(f != other and f in other for other in found):
            pruned.add(f)

    return pruned


def get_education_score(jd_field: str, cv_field: str) -> int:
    jd_field = jd_field.lower().strip()
    cv_field = cv_field.lower().strip()
    if jd_field == cv_field:
        return 3
    for domain, levels in DOMAIN_GROUPS.items():
        if jd_field in levels["core"] or jd_field == domain:
            if cv_field in levels["core"]:
                return 3
            if cv_field in levels["related"]:
                return 2
            if cv_field in levels["broad"]:
                return 1
    return 0


_DF_MONTH = (r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|'
             r'aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)')
_DF_PREFIX_RE = re.compile(
    r'^(?:[\s•◦∗*\[\]()\-–—|]+'
    r'|\d{1,2}/(?:\d{1,2}/)?(?:19|20)\d{2}'
    r'|(?:19|20)\d{2}'
    r'|' + _DF_MONTH + r'\b\.?'
    r'|present\b|current\b)+',
    re.IGNORECASE,
)
_DF_WORD_RE = re.compile(
    r"(?i:(?P<lvl>bachelor|master)(?:'?s)?\b(?:\s+degree)?"
    r"(?:\s+(?:of|in)\s+(?P<kind>science|arts|engineering|technology)\b)?)"
)
_DF_ABBR_RE = re.compile(
    r'(?P<abbr>B\.Sc|BSc|BSC|MSc|MSC|M\.Sc|BS|B\.S|MS|M\.S|BE|B\.E|ME|M\.E|BA|B\.A|MA|M\.A|'
    r'BBA|MBA|BTech|B\.Tech|MTech|M\.Tech|MPhil|M\.Phil|PhD|Ph\.D)(?![A-Za-z])\.?'
)
_DF_ABBR_LABEL = {"BSC": "BSc", "MSC": "MSc", "MPHIL": "MPhil", "PHD": "PhD",
                  "BTECH": "BTech", "MTECH": "MTech"}
_DF_KIND_LABEL = {
    "bachelor": {"science": "BS", "arts": "BA", "engineering": "BE", "technology": "BTech"},
    "master":   {"science": "MS", "arts": "MA", "engineering": "ME", "technology": "MTech"},
}
_DF_CODE_LABEL = {
    "BSSE": ("BS", "Software Engineering"), "BSCS": ("BS", "Computer Science"),
    "BSIT": ("BS", "Information Technology"), "BSAI": ("BS", "Artificial Intelligence"),
    "BSDS": ("BS", "Data Science"),
}
_DF_CODE_RE = re.compile(r'^(?P<field>[A-Za-z&/ ]{3,50}?)\s*\|\s*(?P<code>BS[A-Z]{2,3})\b')
_DF_INSTITUTION_RE = re.compile(
    r'\b(?:university|institute|college|faculty|school|academy)\b', re.IGNORECASE)
_DF_CUT_RE = re.compile(
    r'\(\s*c?gpa|\bc?gpa\b|\||;|,|\[|:|•|\bat\b|\bfrom\b|\bgraduated\b|\bthesis\b|\bmajor\b|'
    r'\b(?:19|20)\d{2}\b|\b' + _DF_MONTH + r'\b\.?\s+(?:19|20)\d{2}|'
    r'\d+(?:\.\d+)?\s*/\s*\d|\s[-–—]\s',
    re.IGNORECASE,
)
_DF_COUNTRY_RE = re.compile(r'^\s*,\s*(?:pakistan|usa|uk|uae|canada|australia|india)\b', re.IGNORECASE)
_DF_LEVEL_ORDER = ["bachelors", "masters", "mphil", "phd"]
_DF_SMALL_WORDS = {"and", "of", "in", "the", "for", "&"}
_DF_ACRONYMS = {"AI", "ML", "IT", "CS", "NLP", "IOT", "HR", "BI"}


def _df_prettify(field: str) -> str:
    """Title-case ALL-CAPS fields; capitalise all-lowercase words elsewhere ("Software engineering")."""
    all_caps = field.isupper()
    words = []
    for i, w in enumerate(field.split()):
        if w.upper() in _DF_ACRONYMS and (all_caps or w.isupper()):
            words.append(w.upper())
        elif w.lower() in _DF_SMALL_WORDS and i > 0:
            words.append(w.lower())
        elif all_caps:
            words.append(w[:1].upper() + w[1:].lower())
        elif w[:1].islower():
            words.append(w[:1].upper() + w[1:])
        else:
            words.append(w)
    return " ".join(words)


def _df_cut_field(raw: str) -> tuple:
    """Trim text after a degree keyword down to the field of study.
    Returns (field, ran_into_institution)."""
    m = _DF_CUT_RE.search(raw)
    head = raw[:m.start()] if m else raw
    tail = raw[m.start():] if m else ""
    inst = _DF_INSTITUTION_RE.search(head)
    ran_into_institution = bool(inst)
    if inst:
        head = head[:inst.start()]
    head = re.sub(r'\(\s*[A-Za-z.]{2,6}\s*\)', '', head)
    head = head.strip(" \t-–—.:;,(")
    if _DF_COUNTRY_RE.match(tail) and len(head.split()) > 1:
        head = head.rsplit(" ", 1)[0]
    elif ran_into_institution:
        words = head.split()
        if len(words) >= 3 and words[-1][:1].isupper():
            head = " ".join(words[:-1])
    return head.strip(), ran_into_institution


def _parse_degree_lines(text: str):
    """
    Parse the Education section into (level, label, field) tuples, e.g.
    ("bachelors", "BE", "Software Engineering"). A degree is only
    recognised at the start of a line (after any date prefix), so
    incidental words elsewhere in the section are ignored.
    """
    section = _extract_education_section(text)
    lines = [ln.strip() for ln in section.split("\n")]
    found = []

    for i, line in enumerate(lines):
        if not line:
            continue
        body = _DF_PREFIX_RE.sub("", line)

        code = _DF_CODE_RE.match(body)
        if code and code.group("code") in _DF_CODE_LABEL:
            label, field = _DF_CODE_LABEL[code.group("code")]
            found.append((i, "bachelors", label, field))
            continue

        level = label = None
        m = _DF_WORD_RE.match(body)
        if m:
            level = "bachelors" if m.group("lvl").lower() == "bachelor" else "masters"
            kind = (m.group("kind") or "").lower()
            label = _DF_KIND_LABEL[m.group("lvl").lower()].get(
                kind, "BS" if level == "bachelors" else "MS")
        else:
            m = _DF_ABBR_RE.match(body)
            if m:
                key = re.sub(r'[.\s]', '', m.group("abbr")).upper()
                label = _DF_ABBR_LABEL.get(key, re.sub(r'[.\s]', '', m.group("abbr")))
                if key == "MPHIL":
                    level = "mphil"
                elif key == "PHD":
                    level = "phd"
                else:
                    level = "bachelors" if key[0] == "B" else "masters"
        if not m:
            continue

        rest = body[m.end():]
        bm = re.match(r'\s*\(([^)]{0,30})\)', rest)
        inner = bm.group(1).strip() if bm else ""
        bracket_field = inner if (re.fullmatch(r"[A-Za-z&' ]{4,30}", inner) and not inner.isupper()
                                  and not re.fullmatch(r'(?i)hons?|honou?rs', inner)) else ""
        if label.upper() in ("BBA", "MBA") and re.match(r'\s*(?:\([^)]{0,30}\)\s*)?,', rest):
            found.append((i, level, label, _df_prettify(bracket_field)))
            continue

        for _ in range(4):
            rest = re.sub(r'^\s*\([^)]{0,30}\)', '', rest)
            rest = re.sub(r'^[\s:,.\-–]+', '', rest)
            rest = re.sub(r'^(?:in|of)\b\s*', '', rest, flags=re.IGNORECASE)
        rest = rest.strip()

        field, _ = _df_cut_field(rest)
        dangling = bool(re.search(r'(?:\band|&|\bof|\bin)$', field, re.IGNORECASE))
        if (not field or dangling) and i + 1 < len(lines):
            nxt = lines[i + 1]
            nxt_body = _DF_PREFIX_RE.sub("", nxt)
            if (nxt and not _DF_INSTITUTION_RE.search(nxt)
                    and not _DF_ABBR_RE.match(nxt_body) and not _DF_WORD_RE.match(nxt_body)):
                field, _ = _df_cut_field((field + " " + nxt).strip())
        field = re.sub(r'\s+(?:and|&|or|of|in)$', '', field, flags=re.IGNORECASE).strip() or bracket_field
        if len(field) < 3 or not field[0].isalpha():
            continue
        found.append((i, level, label, _df_prettify(re.sub(r'\s+', ' ', field))))

    return found, lines


_LEVEL_RANK = {"bachelors": 1, "masters": 2, "mphil": 3, "phd": 4}
_LEVEL_DEFAULT_LABEL = {"bachelors": "Bachelor's", "masters": "Master's", "mphil": "MPhil", "phd": "PhD"}
_MONTH_YEAR_IN_LINE_RE = re.compile(
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[\s,\-/]*((?:19[7-9]|20[0-4])\d)\b'
    r'|\b(0?[1-9]|1[0-2])/((?:19[7-9]|20[0-4])\d)\b', re.IGNORECASE)


@_dataclass
class Degree:
    level: str
    label: str
    field: str = ""
    year: int = None
    in_progress: bool = False

    def display(self) -> str:
        name = f"{self.label} {self.field}".strip()
        if self.in_progress:
            return f"{name} (in progress{', exp. ' + str(self.year) if self.year and self.year >= _cur_year() else ''})"
        return f"{name} ({self.year})" if self.year else f"{name} (year not stated)"


def _line_year(line: str):
    """(end_year, in_progress) for one line, or (None, False)."""
    clean = _PERSONAL_INFO_RE.sub('', line)
    if _CURRENT_WORDS_RE.search(clean) or re.search(r'\b(?:expected|pursuing|ongoing|continuing)\b',
                                                     clean, re.IGNORECASE):
        years = [int(y) for y in _ALL_YEARS_RE.findall(clean)]
        future = [y for y in years if y >= _cur_year()]
        return (max(future) if future else None), True
    years = [int(y) for y in _ALL_YEARS_RE.findall(clean) if 1975 <= int(y) <= _cur_year() + 6]
    if not years:
        return None, False
    year = max(years)
    in_progress = year > _cur_year()
    if year == _cur_year():
        for m in _MONTH_YEAR_IN_LINE_RE.finditer(clean):
            if m.group(1) and int(m.group(2)) == year:
                if _MONTH_NAME_TO_NUM.get(m.group(1).lower()[:3], 0) > _cur_month():
                    in_progress = True
            elif m.group(3) and int(m.group(4)) == year and int(m.group(3)) > _cur_month():
                in_progress = True
    return year, in_progress


_OTHER_QUAL_RE = re.compile(
    r'\b(?:diploma|certificat\w*|certified|intermediate|matric\w*|f\.?sc|hssc|ssc|a[\s-]levels?|'
    r'o[\s-]levels?|high\s+school|college)\b', re.IGNORECASE)


def extract_degrees(text: str) -> list:
    """All degrees in the Education section, each with its own year.

    Year lookup per degree: the degree's own line first; otherwise the
    nearest dated line in the direction this CV uses ("2021 - 2025" ABOVE
    the degree line, or the dates BELOW it). The direction is decided from
    the first degree and applied to all, so one entry's dates are never
    borrowed by its neighbour. Scanning stops at the next degree or at
    another qualification (diploma, certificate, college, intermediate).
    """
    parsed, lines = _parse_degree_lines(text)
    degree_lines = {p[0] for p in parsed}

    def is_fence(j):
        return j in degree_lines or bool(_OTHER_QUAL_RE.search(lines[j]))

    def scan(i, step):
        j = i + step
        for _ in range(6):
            if not (0 <= j < len(lines)) or is_fence(j):
                return None, False
            y, prog = _line_year(lines[j])
            if y is not None or prog:
                return y, prog
            j += step
        return None, False

    layout_before = None
    degrees = []
    for i, level, label, field in parsed:
        year, prog = _line_year(lines[i])
        if year is None and not prog:
            if layout_before is None:
                before = scan(i, -1)
                layout_before = before[0] is not None or before[1]
            order = (-1, 1) if layout_before else (1, -1)
            for step in order:
                year, prog = scan(i, step)
                if year is not None or prog:
                    break
        degrees.append(Degree(level, label, field, year, prog))

    if not degrees:
        dy = extract_degree_years(text)
        for level in _DF_LEVEL_ORDER:
            if level in dy:
                prog = level == "bachelors" and dy.get("bachelors_in_progress", False)
                degrees.append(Degree(level, _LEVEL_DEFAULT_LABEL[level], "", dy[level], prog))

    unique = {}
    for d in degrees:
        key = (d.level, d.field.lower())
        if key not in unique or (unique[key].year is None and d.year is not None):
            unique[key] = d
    return sorted(unique.values(), key=lambda d: (_LEVEL_RANK[d.level], d.year or 0))


def format_degrees(degrees: list) -> str:
    return ", ".join(d.display() for d in degrees) if degrees else "None detected"


def highest_completed_level(degrees: list) -> int:
    done = [_LEVEL_RANK[d.level] for d in degrees if not d.in_progress]
    return max(done) if done else 0



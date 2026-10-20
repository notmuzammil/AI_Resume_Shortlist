"""
Single tuning surface for the whole system. Nothing here should be
duplicated in the app or the modules.

Values that differ per deployment (LLM host/model, upload limits,
password) can be overridden with environment variables.
"""
import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


THRESHOLDS = {
    "Highly Suitable": 75,
    "Suitable": 55,
    "Moderate": 25,
    "Not Suitable": 0,
}

WEIGHTS = {
    "skills": 0.45,
    "experience": 0.32,
    "education": 0.13,
    "title": 0.10,
}

MUST_HAVE_WEIGHT = 2.5
NICE_TO_HAVE_WEIGHT = 0.5

MATCH_CREDIT = {
    "direct": 1.0,
    "alias": 1.0,
    "inferred": 0.9,
    "llm": 0.75,
    "course": 0.5,
    "related": 0.5,
}

INFERENCE_CREDIT = {
    "strong": 0.9,
    "moderate": 0.75,
    "weak": 0.5,
}

MUST_HAVE_MIN_COVERAGE = 0.65
MUST_HAVE_LOW_COVERAGE = 0.35
MUST_HAVE_SCORE_CAP = 45
MUST_HAVE_SCORE_FLOOR_CAP = 30
SOFT_CAP_SPREAD = 0.2
MUST_HAVE_MAX_SHARE = 0.80

EXP_SHORTFALL_EXPONENT = 0.85
EXP_AT_MINIMUM = 0.85
EXP_DEPTH_YEARS = 3.0
EXP_NO_MINIMUM_WEIGHT = 0.5
EXP_NO_MINIMUM_BASE = 0.5

SENIORITY_YEARS = {
    "entry": 0.0,
    "junior": 1.0,
    "mid": 0.0,
    "senior": 5.0,
    "lead": 7.0,
}

OVERQUALIFIED_MARGIN_YEARS = 5.0

BELOW_MIN_EXPERIENCE_POLICY = os.environ.get("CV_BELOW_MIN_EXP", "cap").strip().lower()
BELOW_MIN_EXPERIENCE_CAP = THRESHOLDS["Suitable"] - 1
BELOW_MIN_EXPERIENCE_FLOOR_CAP = THRESHOLDS["Moderate"]
BELOW_MIN_EXPERIENCE_SCALE_YEARS = 2.0
MIN_EXPERIENCE_TOLERANCE_YEARS = 0.1
MIN_EXPERIENCE_TOLERANCE_RATIO = 0.90

MISSING_TITLE_SKILL_CAP = THRESHOLDS["Suitable"] - 1

MAX_FILE_BYTES = _env_int("CV_MAX_FILE_MB", 10) * 1_048_576
MAX_FILES_PER_BATCH = _env_int("CV_MAX_FILES", 200)
MAX_PDF_PAGES = _env_int("CV_MAX_PDF_PAGES", 15)
MAX_DOCX_UNCOMPRESSED = 50 * 1_048_576
MIN_TEXT_CHARS = 80
MAX_CV_TEXT_CHARS = 100_000
MAX_JD_CHARS = 30_000

OCR_ENABLED = _env_bool("CV_OCR_ENABLED", True)
OCR_RENDER_DPI = 200
MAX_OCR_PAGES = _env_int("CV_MAX_OCR_PAGES", 4)
MIN_CHARS_PER_PAGE_BEFORE_OCR = 40

EXTRACTION_WORKERS = _env_int("CV_WORKERS", 4)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_HOST.startswith(("http://", "https://")):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
LLM_PREFERRED_MODELS = ["qwen3:4b", "qwen3:8b", "llama3.1:8b", "qwen2.5:7b", "llama3.2:3b", "llama3.2:1b"]
_env_model = os.environ.get("CV_LLM_MODEL", "").strip()
if _env_model:
    LLM_PREFERRED_MODELS.insert(0, _env_model)
LLM_CLOUD_BASE_URL = os.environ.get("CV_LLM_BASE_URL", "").rstrip("/")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
LLM_CLOUD_DEFAULT_MODELS = {
    NVIDIA_BASE_URL: ["nvidia/nemotron-3-super-120b-a12b", "openai/gpt-oss-20b"],
    OPENAI_BASE_URL: ["gpt-4.1-mini", "gpt-4o-mini"],
}
LLM_CLOUD_READ_TIMEOUT = _env_float("CV_LLM_CLOUD_TIMEOUT", 90.0)
LLM_CLOUD_PARALLEL = _env_int("CV_LLM_CLOUD_PARALLEL", 4)
LLM_CONNECT_TIMEOUT = 3.0
LLM_READ_TIMEOUT = _env_float("CV_LLM_TIMEOUT", 300.0)
LLM_MAX_RETRIES = 1
LLM_CONTEXT_TOKENS = 4096
LLM_MAX_CV_CHARS = 6_000
LLM_MAX_REQUIREMENTS = 6
LLM_PARALLEL = _env_int("CV_LLM_PARALLEL", 1)
LLM_BATCH_BUDGET_SECONDS = _env_float("CV_LLM_BUDGET", 1800.0)
LLM_CIRCUIT_BREAKER_FAILURES = 3
LLM_MIN_DETERMINISTIC_SCORE = 20.0
LLM_MAX_CANDIDATES = _env_int("CV_LLM_MAX_CANDIDATES", 5)

# Cloud AI scoring (modules/ai_evaluator.py): the model judges each requirement,
# code turns the verdicts into the score.
AI_IMPORTANCE_WEIGHT = {"must": 3.0, "preferred": 1.0}
AI_SOFT_SKILL_WEIGHT = 0.5
AI_STATUS_CREDIT = {"met": 1.0, "inferred": 0.6, "partial": 0.5, "not_evidenced": 0.0, "confirmed_gap": 0.0}
AI_SOURCE_CREDIT = {
    "work": 1.0, "certification": 1.0, "internship": 0.85, "project": 0.75,
    "education": 0.6, "skills_list": 0.6, "none": 0.6,
}
AI_ROLE_KIND_WEIGHT = {
    "full_time": 1.0, "freelance": 1.0, "unknown": 1.0, "part_time": 0.5, "internship": 0.5,
    "volunteer": 0.25, "academic_project": 0.0,
}
AI_STATED_YEARS_MAX_CREDIT = 0.75
AI_SECOND_OPINION = _env_bool("CV_AI_SECOND_OPINION", True)
AI_SECOND_OPINION_MARGIN = 5.0


"""
Deterministic skill extraction.

All matching is done against a curated vocabulary plus guarded
discovery of technology-looking tokens. Abbreviations and spellings are
collapsed to ONE canonical form on both the JD and the CV side
("NLP" ⇔ "Natural Language Processing", "Tensor-flow" ⇔ "TensorFlow"),
so they always compare equal.

NOTE: spaCy used to be loaded here (`spacy.load("en_core_web_md")`) but
the model was never used — it cost ~1 GB of dependencies and several
seconds of start-up. It has been removed.
"""
import re

STOPWORDS = {
    "skills", "experience", "mentoring", "knowledge", "ability",
    "understanding", "familiar", "exposure", "proficiency", "expertise",
    "background", "capability", "competency", "working", "strong",
    "good", "excellent", "hands", "years", "year", "month", "months",
    "team", "teams", "work", "job", "role", "position", "candidate",
    "requirement", "requirements", "responsibilities", "responsibility",
    "description", "looking", "seeking", "required", "preferred",
    "plus", "bonus", "etc", "use", "using", "used", "able", "must",
    "need", "needs", "well", "also", "new", "high", "level", "based",
    "related", "field", "area", "areas", "type", "types", "way",
    "best", "great", "key", "core", "main", "real", "set", "sets",
    "time", "day", "days", "week", "project", "projects", "task",
    "tasks", "management", "manager", "lead", "leader", "senior",
    "junior", "intern", "internship", "company", "organization",
    "client", "clients", "customer", "customers", "service", "services",
    "solution", "solutions", "product", "products", "system", "systems",
    "application", "applications", "platform", "platforms", "tool",
    "tools", "process", "processes", "method", "methods", "approach",
    "communication", "collaboration", "problem", "problems", "solving",
    "space", "science", "survey", "data", "other", "with", "gnu",
    "octave", "topcat", "sdss", "wise", "astronomical",
    "developer", "developers", "stack", "stacks", "development", "programmer",
    "engineer", "engineers", "engineering", "designer", "designers",
    "architect", "specialist", "administrator",
    "figure", "table", "section", "chapter", "page", "note", "see",
    "ref", "include", "includes", "including", "such", "like", "per",
    "via", "both", "each", "their", "have", "has", "had", "been",
    "being", "will", "would", "could", "should", "may", "might",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
    "nov", "dec", "present", "current",
    "or", "and", "not", "but", "for", "from", "into", "onto",
    "over", "under", "up", "down", "in", "out", "on", "off",
    "cv", "pdf", "doc", "ltd", "inc", "pvt", "llc", "llp", "co",
    "usa", "uk", "uae", "ksa", "pk", "eu", "dob", "cnic", "nic",
    "gpa", "cgpa", "phone", "email", "www", "com", "org", "net",
    "bsc", "msc", "bs", "ms", "be", "me", "phd", "mba", "bba", "llb",
    "mphil", "matric", "fsc", "ics", "hssc", "sscbe",
    "education", "profile", "objective", "summary", "certifications",
    "certification", "awards", "achievements", "languages",
    "references", "reference", "interests", "hobbies", "contact",
    "address", "nationality", "gender", "linkedin", "github.com",
    "about", "skill", "tech", "the", "we", "you", "our", "your",
    "id", "fyp", "ceo", "cto", "hr", "pm", "am", "ok", "na",
    "it", "ict", "imcs", "muet", "smiu", "iba", "nust", "fast", "smit",
    "ide", "ui", "ux", "cs", "se", "ee", "ce", "e.g", "e.g.", "i.e", "i.e.", "etc.",
    "faq", "asap", "eta", "kpi", "kpis", "roi", "b2b", "b2c",
    "ideal", "what", "who", "why", "how", "join", "apply", "offer", "perks", "duty",
    "title", "basic", "this", "that", "are", "can", "all", "get", "now", "open", "roles", "jobs",
}

SHORT_SKILL_ALLOWLIST = {"ai", "ml", "js", "ts", "qa", "bi", "c", "r", "go"}
_SPECIAL_ONLY = {"c", "r", "go"}

NOISE_PATTERNS = re.compile(
    r'^\d+$'
    r'|^\d{4}$'
    r'|^\d+[\.\-]\d+$'
    r'|^[a-z]{1,2}$'
    r'|^(https?|www)\b'
)

SLASH_STOPWORDS = {
    "and/or", "i/o", "b/w", "r/w", "n/a", "w/o", "w/",
    "mr/ms", "he/she", "him/her", "his/her",
}

KNOWN_SKILLS = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#",
    "ruby", "php", "swift", "kotlin", "go", "rust", "scala", "r",
    "matlab", "perl", "bash", "shell", "powershell", "golang", "dart",
    "spring boot", "spring mvc", "spring security", "spring",
    "hibernate", "jpa", "maven", "gradle", "junit", "j2ee", "jakarta ee",
    "tomcat", "wildfly", "swagger", "oauth", "oauth2", "jwt", "websocket",
    "kafka", "microservices", "soa", "soap", "webhooks",

    "html", "css", "react", "angular", "vue", "nodejs",
    "django", "flask", "fastapi", "laravel", "express",
    "jquery", "bootstrap", "tailwind", "rest api", "graphql",
    "next.js", "nuxt.js", "webpack", "json", "ajax",

    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data science", "data analysis", "data analytics",
    "data engineering", "big data", "data mining", "data visualization",
    "artificial intelligence", "neural networks", "transformers",
    "reinforcement learning", "feature engineering", "predictive modeling",
    "statistical modeling", "time series", "transfer learning",
    "supervised learning", "unsupervised learning", "classification",
    "regression", "clustering", "statistics", "pattern recognition",
    "generative ai", "llm", "cnn", "rnn", "lstm", "gan", "autoencoders",
    "sentiment analysis", "named entity recognition", "text classification",
    "chatbot", "chatbots", "speech recognition", "recommendation systems",
    "ocr", "face recognition",

    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
    "huggingface", "spacy", "nltk", "opencv", "xgboost", "lightgbm",
    "catboost", "yolo", "bert", "gpt", "stable diffusion",
    "langchain", "llama", "fastai", "onnx", "tensorrt",

    "image processing", "object detection", "image segmentation",
    "image classification", "optical flow", "3d reconstruction",

    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
    "sqlite", "cassandra", "elasticsearch", "firebase", "dynamodb",
    "nosql", "neo4j", "influxdb",
    "sql server", "pl/sql", "mariadb", "couchbase",
    "snowflake", "bigquery", "redshift", "stored procedures",
    "query optimization", "database administration", "dba",
    "data warehousing", "replication", "sharding",
    "database design", "etl",

    "flutter", "react native", "swiftui", "jetpack compose",
    "objective-c", "ionic", "xamarin", "android sdk", "cocoapods",
    "android", "ios", "mvc", "mvvm", "mvp", "clean architecture",
    "state management", "bloc", "getx", "riverpod", "provider",

    "aws", "azure", "google cloud", "docker", "kubernetes", "openstack",
    "jenkins", "git", "github", "gitlab", "ci/cd", "terraform",
    "ansible", "linux", "unix", "devops", "mlops", "airflow", "ssh",

    "jupyter", "vs code", "pycharm", "intellij",
    "android studio", "xcode", "postman", "jira", "confluence",
    "excel", "power bi", "tableau", "sap", "erp", "figma",
    "adobe xd", "photoshop", "illustrator",

    "embedded systems", "arduino", "raspberry pi", "fpga",
    "microcontroller", "teensy", "rtos", "vhdl", "verilog",
    "iot", "robotics", "ros",

    "astropy",

    "agile", "scrum", "kanban", "tdd", "bdd", "api development",

    "project management", "leadership", "teamwork",
    "problem solving", "critical thinking", "time management",

    "streamlit", "gradio", "openai", "anthropic", "gemini", "claude",
    "ollama", "vllm", "mistral", "llamaindex",
    "rag", "vector database", "vector search", "embeddings", "pinecone",
    "chromadb", "faiss", "weaviate", "qdrant", "milvus", "pgvector", "lancedb",
    "prompt engineering", "fine-tuning", "lora", "peft",
    "quantization", "diffusers", "whisper", "langgraph", "llamafile",
    "semantic search", "ai agents", "function calling", "mcp", "n8n",

    "svelte", "vite", "nestjs",
    "redux", "zustand", "tanstack", "shadcn", "material ui",
    "sass", "less", "storybook", "trpc", "prisma", "drizzle",
    "supabase", "appwrite", "socket.io", "grpc", "protobuf",
    "rabbitmq", "celery", "nginx", "apache", "serverless",
    "mern", "mean", "lamp", "jamstack", "full stack",

    "jest", "vitest", "cypress", "playwright", "selenium", "appium",
    "pytest", "unittest", "mocha", "testing library",

    "github actions", "gitlab ci", "circleci", "travis ci", "argocd",
    "helm", "prometheus", "grafana", "datadog", "sentry", "splunk",
    "cloudformation", "pulumi", "vagrant", "openshift", "svn",
    "bitbucket", "mercurial",

    "mlflow", "kubeflow", "dvc", "wandb",
    "dbt", "spark", "pyspark", "hadoop", "hive", "flink", "databricks",
    "sagemaker", "vertex ai", "azure ml", "great expectations",
    "polars", "duckdb",

    "account management", "key account management", "sales", "business development",
    "lead generation", "negotiation", "crm", "salesforce", "hubspot", "customer service",
    "customer relationship management", "marketing", "digital marketing", "seo", "sem",
    "social media marketing", "content writing", "copywriting", "market research",
    "public relations", "event planning", "recruitment", "talent acquisition", "payroll",
    "onboarding", "employee relations", "performance management", "bookkeeping", "accounting",
    "quickbooks", "budgeting", "forecasting", "financial analysis", "auditing", "taxation",
    "supply chain management", "procurement", "inventory management", "logistics",
    "hris", "ats", "workday", "bamboohr", "successfactors", "compensation and benefits", "labour law",
    "employer branding", "boolean search", "linkedin recruiter", "employee engagement",
    "training and development", "succession planning", "talent management", "hr operations",
    "hr policies", "ms office", "ms word", "powerpoint", "upselling", "customer success",

    "backend development", "frontend development", "distributed systems", "system design",
    "performance optimization", "software architecture",
    "mobile development", "cross-platform development", "unit testing", "test automation", "software testing",
    "infrastructure as code", "containerization", "orm", "data pipelines", "business intelligence",
    "message queues", "event-driven architecture", "ui/ux", "financial reporting",

    "sqlalchemy", "sequelize", "typeorm", "mongoose", "codeigniter", "asp.net", "looker", "qlik",
    "crewai", "autogen", "xero",
}

SKILL_ALIASES = {
    "js": "javascript", "ecmascript": "javascript", "es6": "javascript",
    "ts": "typescript",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv2": "opencv", "open cv": "opencv",
    "k8s": "kubernetes",
    "postgres": "postgresql", "psql": "postgresql", "postgre sql": "postgresql",
    "node": "nodejs", "node.js": "nodejs", "node js": "nodejs",
    "express.js": "express", "expressjs": "express", "express js": "express",
    "react.js": "react", "reactjs": "react", "react js": "react",
    "vue.js": "vue", "vuejs": "vue", "angularjs": "angular", "angular.js": "angular",
    "nextjs": "next.js", "next js": "next.js", "nuxtjs": "nuxt.js",
    "nest.js": "nestjs",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "scikitlearn": "scikit-learn",
    "hugging face": "huggingface", "hf": "huggingface",
    "tf": "tensorflow", "tensor-flow": "tensorflow", "tensor flow": "tensorflow",
    "torch": "pytorch", "py torch": "pytorch",
    "gcp": "google cloud", "google cloud platform": "google cloud",
    "amazon web services": "aws",
    "ms sql": "sql server", "mssql": "sql server", "microsoft sql server": "sql server",
    "plsql": "pl/sql",
    "ci cd": "ci/cd", "cicd": "ci/cd",
    "restful api": "rest api", "restful apis": "rest api", "rest apis": "rest api",
    "restful": "rest api", "apis": "rest api", "api": "rest api",
    "chroma": "chromadb", "chroma db": "chromadb",
    "llama index": "llamaindex", "llama-index": "llamaindex",
    "lang-chain": "langchain", "lang chain": "langchain",
    "fine tuning": "fine-tuning", "finetuning": "fine-tuning",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "weights and biases": "wandb", "weights & biases": "wandb", "w&b": "wandb",
    "mui": "material ui",
    "scss": "sass",
    "large language models": "llm", "large language model": "llm", "llms": "llm",
    "gen ai": "generative ai", "genai": "generative ai",
    "vector databases": "vector database", "vector db": "vector database",
    "vector store": "vector database", "vector stores": "vector database",
    "neural network": "neural networks", "ann": "neural networks", "anns": "neural networks",
    "convolutional neural networks": "cnn", "convolutional neural network": "cnn", "cnns": "cnn",
    "recurrent neural networks": "rnn", "rnns": "rnn",
    "transformer": "transformers",
    "embedding": "embeddings",
    "jupyter notebook": "jupyter", "jupyter notebooks": "jupyter",
    "visual studio code": "vs code", "vscode": "vs code",
    "ms excel": "excel", "microsoft excel": "excel",
    "powerbi": "power bi",
    "objective c": "objective-c",
    "rest": "rest api",
    "fullstack": "full stack", "full-stack": "full stack",
    "golang": "go",
    "spring framework": "spring",
    "github actions": "github actions",
    "mongo": "mongodb", "mongo db": "mongodb",
    "dynamo db": "dynamodb",
    "fast api": "fastapi",
    "chat gpt": "openai", "chatgpt": "openai",
    "tailwind css": "tailwind", "tailwindcss": "tailwind",
    "j2ee": "j2ee", "java ee": "j2ee",
    "object oriented programming": "oop", "object-oriented programming": "oop",
    "optical character recognition": "ocr",
    "facial recognition": "face recognition",
    "state-management": "state management",
    "ai agent": "ai agents", "agentic ai": "ai agents",
    "sdks": "sdk",
    "customer relationship management": "crm",
    "key account management": "account management", "key accounts": "account management",
    "client relationship management": "account management",
    "sales growth": "sales", "b2b sales": "sales",
    "search engine optimization": "seo", "search engine optimisation": "seo",
    "talent acquisition": "recruitment",
    "recruiting": "recruitment", "full-cycle recruiting": "recruitment", "full cycle recruiting": "recruitment",
    "human resource information system": "hris", "human resources information system": "hris",
    "applicant tracking system": "ats", "applicant tracking systems": "ats",
    "bamboo hr": "bamboohr", "sap successfactors": "successfactors",
    "compensation & benefits": "compensation and benefits", "comp & ben": "compensation and benefits",
    "labor law": "labour law", "labor laws": "labour law", "labour laws": "labour law",
    "employment law": "labour law",
    "learning and development": "training and development", "learning & development": "training and development",
    "training & development": "training and development",
    "microsoft office": "ms office", "ms-office": "ms office", "microsoft word": "ms word",
    "ms powerpoint": "powerpoint", "microsoft powerpoint": "powerpoint",
    "up-selling": "upselling", "upsell": "upselling", "upsells": "upselling", "up-sell": "upselling",
    "backend": "backend development", "back-end": "backend development", "back end": "backend development",
    "back-end development": "backend development", "back end development": "backend development",
    "server-side development": "backend development", "server side development": "backend development",
    "frontend": "frontend development", "front-end": "frontend development", "front end": "frontend development",
    "front-end development": "frontend development", "front end development": "frontend development",
    "distributed system": "distributed systems", "distributed computing": "distributed systems",
    "systems design": "system design", "software design and architecture": "software architecture",
    "performance optimisation": "performance optimization", "performance tuning": "performance optimization",
    "data warehouse": "data warehousing", "data warehouses": "data warehousing",
    "mobile app development": "mobile development", "mobile application development": "mobile development",
    "mobile apps development": "mobile development",
    "cross platform": "cross-platform development", "cross-platform": "cross-platform development",
    "cross platform development": "cross-platform development",
    "unit test": "unit testing", "unit tests": "unit testing",
    "automation testing": "test automation", "automated testing": "test automation",
    "qa testing": "software testing", "manual testing": "software testing", "quality assurance testing": "software testing",
    "iac": "infrastructure as code",
    "containerisation": "containerization",
    "object relational mapping": "orm", "object-relational mapping": "orm", "orms": "orm",
    "data pipeline": "data pipelines",
    "bi": "business intelligence", "bi tools": "business intelligence",
    "message queue": "message queues", "message broker": "message queues", "message brokers": "message queues",
    "message queuing": "message queues",
    "event driven architecture": "event-driven architecture", "event-driven": "event-driven architecture",
    "event driven": "event-driven architecture",
    "ui ux": "ui/ux", "ux/ui": "ui/ux", "ui/ux design": "ui/ux", "ux design": "ui/ux", "ui design": "ui/ux",
    "user experience design": "ui/ux",
    "crew ai": "crewai", "code igniter": "codeigniter", "asp.net core": "asp.net",
}

SOFT_SKILLS = {"problem solving", "critical thinking", "time management", "teamwork",
               "leadership", "communication", "presentation"}

_UNSAFE_TEXT_ALIASES = {"node", "hf", "ts", "dl", "ann", "anns", "api", "apis", "transformer",
                        "embedding", "provider", "torch", "mongo", "restful", "rest", "bi", "iac",
                        "event-driven", "event driven", "cross platform", "cross-platform"}
_TEXT_ALIASES = {k: v for k, v in SKILL_ALIASES.items()
                 if k not in KNOWN_SKILLS and k not in _UNSAFE_TEXT_ALIASES}

AMBIGUOUS_SKILLS = {
    "express", "excel", "swift", "spring", "less", "shell", "oracle",
    "apache", "dart", "ionic", "mocha", "jest", "hive", "helm", "sentry",
    "provider", "bloc", "flask", "gemini", "claude", "mistral", "llama",
    "whisper", "spark", "kafka", "redis", "replication", "sharding",
    "regression", "classification", "clustering", "statistics", "leadership",
    "teamwork", "agile", "embeddings", "transformers", "serverless", "unix",
    "ros", "sap", "erp", "anthropic", "polars", "celery", "drizzle", "prisma",
    "sales", "marketing", "logistics", "onboarding", "forecasting", "budgeting", "procurement",
}

_NOT_REST_LOOKBEHIND = ''.join(
    r'(?<!' + w + r'\s)' for w in ("context", "composition", "canvas", "speech", "audio", "storage",
                                   "reflection", "stream", "collections", "geolocation", "clipboard",
                                   "notification", "vibration", "battery", "options", "dom"))

_SPECIAL_PATTERNS = {
    "rest api": re.compile(
        r'(?i:\b(?:rest(?:ful)?\s*(?:apis?|services?|web\s*services?|endpoints?)|' + _NOT_REST_LOOKBEHIND + r'apis?)\b)'
        r'|(?<![A-Za-z])REST(?![A-Za-z])'),
    "rag": re.compile(r'\bRAG\b|\bretrieval[\s-]*augmented(?:[\s-]*generation)?\b', re.IGNORECASE),
    "ci/cd": re.compile(r'\bci\s*/\s*cd\b|\bci[\s-]?cd\b|\bcontinuous\s+(?:integration|delivery|deployment)\b',
                        re.IGNORECASE),
    "full stack": re.compile(r'\bfull[\s-]?stack\b', re.IGNORECASE),
    "go": re.compile(r'\bGolang\b|\bGo\s+(?:language|lang|programming)\b|\bGo(?=\s*[,/)]\s*(?:Rust|Java|Python|C\+\+))'),
    "c": re.compile(r'(?<![A-Za-z+#])C(?=\s*(?:/\s*C\+\+|,\s*C\+\+|\s+programming|\s+language))'),
    "r": re.compile(r'(?<![A-Za-z])R(?=\s*(?:programming|language|studio|,\s*(?:Python|SQL|MATLAB)))'
                    r'|(?<=Python,\s)R(?![A-Za-z])|(?<=Python\sor\s)R(?![A-Za-z])|\bRStudio\b'),
}
_CAPITALISED_SPRING_RE = re.compile(r'\bSpring\b(?!\s*(?:\'?\d{2,4}|semester|term|break|session|season))')

_TECH_LIST_HINT = re.compile(r'[,/|•·]')


def canonicalize_skill(skill: str) -> str:
    s = re.sub(r'\s+', ' ', skill.strip().lower())
    return SKILL_ALIASES.get(s, s)


def canonicalize(skills) -> set:
    """Map every skill to its canonical form and drop the duplicates."""
    return {canonicalize_skill(s) for s in skills}


_INTERNAL_CAPS_RE = re.compile(r'^[A-Za-z][a-z0-9.]*[A-Z]')
_TECH_PUNCT_RE = re.compile(r'[A-Za-z0-9][+#]|[A-Za-z0-9][./][A-Za-z0-9]')
_ALNUM_MIX_RE = re.compile(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9.\-]+$')
_ACRONYM_RE = re.compile(r'^[A-Z]{2,5}$')


def _looks_technical(token: str) -> bool:
    """True if a token outside the vocabulary still looks like technology."""
    if token.lower() in KNOWN_SKILLS or token.lower() in SKILL_ALIASES:
        return True
    if _TECH_PUNCT_RE.search(token):
        return True
    if _ALNUM_MIX_RE.match(token):
        return True
    if _ACRONYM_RE.match(token):
        return True
    if _INTERNAL_CAPS_RE.match(token) and not token.isupper():
        return True
    return False


def _is_noise(token: str) -> bool:
    t = token.strip().lower()
    if t in SHORT_SKILL_ALLOWLIST:
        return False
    return (
        not t
        or t in STOPWORDS
        or bool(NOISE_PATTERNS.match(t))
        or len(t) < 2
        or len(t.split()) > 4
        or bool(re.match(r'^\d', t))
        or bool(re.search(r'\d{3,}', t))
    )


def _is_react_skill(text: str) -> bool:
    """True if react is used as a technology/framework, not an English verb."""
    if re.search(r'\b(?:react\.?js|reactjs|react-native|react\s+native|react\s+developer|'
                 r'react\s+framework|react\s+frontend|react\s+ui|react\s+hooks?)\b', text, re.IGNORECASE):
        return True
    for m in re.finditer(r'\breact\b', text, re.IGNORECASE):
        start = max(0, m.start() - 15)
        before = text[start:m.start()].lower()
        after = text[m.end():min(len(text), m.end() + 25)].lower()
        if re.search(r'\b(?:to|will|can|could|must|and|shall|should|would|quickly|i)\s+$', before):
            continue
        if re.search(r'^\s*(?:to|with|punctually|quickly|promptly|defensively|immediately|'
                     r'appropriately|in\s+response|under\s+pressure|against|towards)\b', after):
            continue
        return True
    return False


def _line_at(text: str, pos: int) -> str:
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return text[s:e if e != -1 else len(text)]


_BUSINESS_AMBIGUOUS = {"sales", "marketing", "logistics", "onboarding", "forecasting", "budgeting", "procurement"}
_BUSINESS_CONTEXT_RE = re.compile(
    r'\b(?:account\s+management|business\s+development|customer\s+(?:success|service)|crm|negotiation|'
    r'lead\s+generation|recruitment|payroll|hris|upsell\w*|cross[\s-]?sell\w*|'
    r'(?:sales|marketing)\s+(?:targets?|goals?|quotas?|teams?|strateg\w+|plans?|pipelines?|campaigns?|'
    r'experience|operations|growth|performance|revenue)|'
    r'(?:experience|background|career|years?)\s+in\s+(?:\w+\s+){0,3}(?:sales|marketing|logistics|procurement))\b',
    re.IGNORECASE)


def _ambiguous_ok(text: str, start: int, end: int, skill: str) -> bool:
    """Accept an ambiguous-word hit only in a technology context."""
    surface = text[start:end]
    if skill == "spring":
        return bool(_CAPITALISED_SPRING_RE.match(text, start))
    if surface[:1].isupper():
        before = text[max(0, start - 4):start].lower()
        if re.search(r'\b(?:to|i)\s+$', before) and skill in ("express", "excel"):
            return False
        return True
    line = _line_at(text, start)
    if skill in _BUSINESS_AMBIGUOUS:
        return bool(_BUSINESS_CONTEXT_RE.search(line))
    hits = 0
    low = line.lower()
    for other in ("python", "java", "sql", "html", "css", "react", "node", "django",
                  "docker", "git", "aws", "mongodb", "mysql", "javascript", "flutter",
                  "tensorflow", "pytorch", "pandas", "numpy", "linux", "kotlin", "php"):
        if other != skill and re.search(r'(?<![a-z])' + re.escape(other) + r'(?![a-z])', low):
            hits += 1
    return hits >= 1 and bool(_TECH_LIST_HINT.search(line))


_VOCAB = sorted(set(KNOWN_SKILLS) | set(_TEXT_ALIASES), key=len, reverse=True)
_VOCAB_RE = {
    s: re.compile(r'(?<![A-Za-z\d])' + re.escape(s).replace(r'\ ', r'[\s\-]+')
                  + r'(?:[\s-]?v?\d+(?:\.\d+)*(?![A-Za-z]))?(?![A-Za-z\d+#])'
                  + (r'(?!-idf)' if s == "tf" else ''),
                  re.IGNORECASE)
    for s in _VOCAB
    if s not in _SPECIAL_PATTERNS and s not in ("c", "r", "go", "react")
}


def skill_mentions(text: str) -> list:
    """
    Positions of vocabulary skills in `text`:
    [(canonical_skill, start, end), ...] sorted by position, with
    overlapping shorter hits removed ("spring boot" wins over "spring").
    """
    raw = []
    for skill, pat in _VOCAB_RE.items():
        canon = canonicalize_skill(skill)
        for m in pat.finditer(text):
            if skill in AMBIGUOUS_SKILLS:
                if not _ambiguous_ok(text, m.start(), m.end(), skill):
                    continue
            raw.append((canon, m.start(), m.end()))
    for skill, pat in _SPECIAL_PATTERNS.items():
        for m in pat.finditer(text):
            raw.append((skill, m.start(), m.end()))
    for m in re.finditer(r'\breact(?:\.?js)?\b|\breact[\s-]native\b', text, re.IGNORECASE):
        surface = m.group(0).lower()
        if "native" in surface:
            raw.append(("react native", m.start(), m.end()))
        elif _is_react_skill(text[max(0, m.start() - 20):m.end() + 30]):
            raw.append(("react", m.start(), m.end()))

    raw.sort(key=lambda r: (r[1], -(r[2] - r[1])))
    kept = []
    max_end = -1
    for canon, s, e in raw:
        if e <= max_end:
            continue
        kept.append((canon, s, e))
        max_end = e
    return kept


_HYPHEN_SUFFIX_RE = re.compile(r'^(.+?)-(?:based|powered|driven|native|related|assisted|enabled|first|'
                               r'ready|focused|oriented|specific|level|like|centric|heavy|savvy|style)$')


def _discovery_skip(token_lower: str) -> bool:
    """Vocabulary words are found (with context guards) by skill_mentions();
    the free-form discovery steps must not re-admit them unguarded."""
    t = re.sub(r'[\s-]?v?\d+(?:\.\d+)*$', '', token_lower) or token_lower
    return (t in AMBIGUOUS_SKILLS or token_lower in AMBIGUOUS_SKILLS or t in _SPECIAL_ONLY
            or t in KNOWN_SKILLS or t in SKILL_ALIASES or token_lower in SKILL_ALIASES
            or bool(_HYPHEN_SUFFIX_RE.match(token_lower)) or len(token_lower.split()) > 2)


_BRACKET_LEAD_RE = re.compile(r'^(?:e\.?\s?g\.?|i\.?\s?e\.?|such\s+as|like|incl(?:uding|\.)?|for\s+example)'
                              r'\s*[:,]?\s*', re.IGNORECASE)


def extract_skills(text: str, infer: bool = True) -> set:
    """All canonical skills found in `text` (CV or JD).
    infer=False skips composite-stack inference (MERN, full stack...)."""
    skills = {canon for canon, _, _ in skill_mentions(text)}

    for word in re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:[.\-][a-zA-Z0-9]+)*\+*#?', text):
        if '/' in word:
            continue
        w = word.lower()
        if _discovery_skip(w):
            continue
        if len(word) >= 2 and not _is_noise(w) and _looks_technical(word):
            skills.add(w)

    for item in re.findall(r'\b\w[\w.+#]*\/\w[\w.+#]*\b', text):
        if item.lower() in SLASH_STOPWORDS:
            continue
        parts = item.split('/')
        if all(len(p) > 1 for p in parts):
            for part in parts:
                p = part.strip().lower()
                if not _discovery_skip(p) and not _is_noise(p) and _looks_technical(part.strip()):
                    skills.add(p)

    for group in re.findall(r'\(([^)]{2,80})\)', text):
        if sum(c.isdigit() for c in group) / max(len(group), 1) > 0.4:
            continue
        for item in re.split(r'[,;/]', group):
            item = _BRACKET_LEAD_RE.sub('', item.strip())
            if not _discovery_skip(item.lower()) and not _is_noise(item.lower()) and _looks_technical(item):
                skills.add(item.lower())

    cleaned = {s.strip() for s in skills if not _is_noise(s)}
    canon = canonicalize(cleaned)
    return infer_composite_stacks(canon) if infer else canon


RAG_RETRIEVAL_SKILLS = frozenset({"vector search", "vector database", "pinecone", "chromadb", "faiss", "qdrant",
                                  "weaviate", "milvus", "pgvector", "lancedb"})
RAG_LLM_SKILLS = frozenset({"llm", "langchain", "llamaindex", "langgraph", "openai", "gemini", "claude", "gpt",
                            "llama", "mistral", "ollama"})


def infer_composite_stacks(skills: set) -> set:
    """MongoDB + Express + React + Node → MERN, etc."""
    inferred = set(skills)
    has_mongo = "mongodb" in inferred or "mongoose" in inferred
    has_express = "express" in inferred
    has_react = bool({"react", "react native"} & inferred)
    has_node = "nodejs" in inferred
    has_angular = "angular" in inferred
    if has_mongo and has_express and has_react and has_node:
        inferred.add("mern")
    if has_mongo and has_express and has_angular and has_node:
        inferred.add("mean")
    if ("linux" in inferred and "apache" in inferred and bool({"mysql", "mariadb"} & inferred)
            and bool({"php", "python", "perl"} & inferred)):
        inferred.add("lamp")
    if RAG_RETRIEVAL_SKILLS & inferred and RAG_LLM_SKILLS & inferred:
        inferred.add("rag")
    has_frontend = bool({"react", "vue", "angular", "next.js", "nuxt.js", "svelte"} & inferred) or \
        {"html", "css", "javascript"} <= inferred
    has_backend = bool({"nodejs", "django", "flask", "fastapi", "spring boot", "laravel", "express",
                        "php", "nestjs", "spring"} & inferred)
    has_db = bool({"sql", "mysql", "postgresql", "mongodb", "sqlite", "redis", "oracle", "sql server",
                   "nosql", "firebase", "supabase"} & inferred)
    if has_frontend and has_backend and has_db:
        inferred.add("full stack")
    return inferred
